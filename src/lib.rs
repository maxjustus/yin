mod errors;

use errors::UnknownValueError;

#[derive(Clone, Debug)]
pub struct Yin {
    threshold: f64,
    tau_max: usize,
    tau_min: usize,
    sample_rate: usize,
}

#[derive(Debug)]
pub struct PitchResult {
    pub frequency: f64,
    pub clarity: f64,
}

impl Yin {
    pub fn init(threshold: f64, freq_min: f64, freq_max: f64, sample_rate: usize) -> Yin {
        let tau_max = (sample_rate as f64 / freq_min).floor() as usize;
        let tau_min = (sample_rate as f64 / freq_max).floor() as usize;
        Yin {
            threshold,
            tau_max,
            tau_min,
            sample_rate,
        }
    }

    pub fn estimate_freq(
        &self,
        audio_sample: &[f64],
    ) -> Result<PitchResult, Box<dyn std::error::Error>> {
        let (sample_frequency, clarity) = compute_sample_frequency(
            audio_sample,
            self.tau_min,
            self.tau_max,
            self.sample_rate,
            self.threshold,
        );

        if sample_frequency.is_infinite() || sample_frequency.is_nan() {
            Err(Box::new(UnknownValueError {}))
        } else {
            Ok(PitchResult {
                frequency: sample_frequency,
                clarity,
            })
        }
    }
}

fn diff_function(audio_sample: &[f64], tau_max: usize) -> Vec<f64> {
    let len = audio_sample.len();
    let tau_max = std::cmp::min(len / 2, tau_max);
    let mut diff = vec![0.0; tau_max + 1];

    for tau in 0..=tau_max {
        let mut sum = 0.0;
        for j in 0..(len - tau) {
            let tmp = audio_sample[j] - audio_sample[j + tau];
            sum += tmp * tmp;
        }
        diff[tau] = sum;
    }
    diff
}

fn cmndf(diff: &[f64]) -> Vec<f64> {
    let mut cmndf = vec![0.0; diff.len()];
    cmndf[0] = 1.0;
    let mut running_sum = 0.0;

    for tau in 1..diff.len() {
        running_sum += diff[tau];
        if running_sum == 0.0 {
            cmndf[tau] = 1.0;
        } else {
            cmndf[tau] = diff[tau] * (tau as f64) / running_sum;
        }
    }

    cmndf
}

fn calculate_clarity(diff_fn: &[f64], tau: usize, window_size: usize) -> f64 {
    let start = tau.saturating_sub(window_size);
    let end = std::cmp::min(tau + window_size, diff_fn.len());

    if end <= start {
        return 0.0;
    }

    let min_val = diff_fn[tau];
    let mut mean_surrounding = 0.0;
    let mut count = 0;

    // Calculate mean of surrounding values, excluding the dip
    for i in start..end {
        if i != tau {
            mean_surrounding += diff_fn[i];
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    mean_surrounding /= count as f64;

    // Calculate clarity based on the ratio of the dip to surrounding values
    let contrast = (mean_surrounding - min_val) / mean_surrounding;

    // Apply sigmoid-like function to normalize clarity between 0 and 1
    // and make it more sensitive to noise
    let clarity = 1.0 / (1.0 + (-10.0 * (contrast - 0.5)).exp());

    clarity
}

fn compute_diff_min(
    diff_fn: &[f64],
    min_tau: usize,
    max_tau: usize,
    harm_threshold: f64,
) -> (f64, f64) {
    let len = diff_fn.len();
    let max_tau = std::cmp::min(max_tau, len - 1);

    let mut tau = min_tau;
    while tau <= max_tau {
        if diff_fn[tau] < harm_threshold {
            while tau < max_tau && diff_fn[tau + 1] < diff_fn[tau] {
                tau += 1;
            }
            let clarity = calculate_clarity(diff_fn, tau, 5); // Use a window of 5 samples
            return (parabolic_interpolation(tau, diff_fn), clarity);
        }
        tau += 1;
    }

    let (tau_min, _) = diff_fn
        .iter()
        .enumerate()
        .skip(min_tau)
        .take(max_tau - min_tau + 1)
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    let clarity = calculate_clarity(diff_fn, tau_min, 5);
    (parabolic_interpolation(tau_min, diff_fn), clarity)
}

fn parabolic_interpolation(tau: usize, diff_fn: &[f64]) -> f64 {
    if tau == 0 || tau >= diff_fn.len() - 1 {
        return tau as f64;
    }
    let s0 = diff_fn[tau - 1];
    let s1 = diff_fn[tau];
    let s2 = diff_fn[tau + 1];

    let denom = s0 + s2 - 2.0 * s1;
    if denom == 0.0 {
        return tau as f64;
    }

    let delta = 0.5 * (s0 - s2) / denom;
    (tau as f64) + delta
}

fn convert_to_frequency(sample_period: f64, sample_rate: usize) -> f64 {
    sample_rate as f64 / sample_period
}

pub fn compute_sample_frequency(
    audio_sample: &[f64],
    tau_min: usize,
    tau_max: usize,
    sample_rate: usize,
    threshold: f64,
) -> (f64, f64) {
    let diff_fn = diff_function(audio_sample, tau_max);
    let cmndf = cmndf(&diff_fn);
    let (sample_period, clarity) = compute_diff_min(&cmndf, tau_min, tau_max, threshold);
    if sample_period <= 0.0 {
        (f64::INFINITY, 0.0)
    } else {
        (convert_to_frequency(sample_period, sample_rate), clarity)
    }
}

#[cfg(test)]
mod tests {
    use dasp::{signal, Signal};

    fn assert_within_tolerance(a: PitchResult, b: f64, tolerance: f64) {
        println!("expected: {}, actual: {}", a.frequency, b);
        assert!((a.frequency - b).abs() < tolerance);
    }

    fn produce_sample(sample_rate: usize, frequency: f64, noise_ratio: f64) -> Vec<f64> {
        use rand::prelude::*;
        let mut rng = thread_rng();
        let mut signal = signal::rate(sample_rate as f64).const_hz(frequency).sine();
        let sample: Vec<f64> = (0..sample_rate)
            .map(|_| signal.next() + noise_ratio * rng.gen::<f64>())
            .collect();
        sample
    }
    use super::*;
    #[test]
    fn sanity_basic_sine() {
        let sample = produce_sample(12, 4.0, 0.0);
        let yin = Yin::init(0.1, 2.0, 5.0, 12);
        let computed_frequency = yin.estimate_freq(&sample).unwrap();
        assert_within_tolerance(computed_frequency, 4.0, 1.0);
    }

    #[test]
    fn strong_harmonic() {
        let sample = produce_sample(44100, 400.0, 0.1);
        let sample2 = produce_sample(44100, 800.0, 0.0);
        let sample3 = produce_sample(44100, 633.3, 0.0);
        let mut combined_sample = vec![];
        for i in 0..sample.len() {
            combined_sample.push((sample[i] * 0.4) + (sample2[i] * 0.6) + (sample3[i] * 0.1));
        }

        let yin = Yin::init(0.1, 300.0, 1000.0, 44100);
        let computed_frequency = yin.estimate_freq(&combined_sample).unwrap();
        assert_within_tolerance(computed_frequency, 400.0, 1.0);
    }

    #[test]
    fn sanity_low_hz_full_sample() {
        let sample = produce_sample(44100, 20.0, 0.0);
        let yin = Yin::init(0.1, 10.0, 100.0, 44100);
        let computed_frequency = yin.estimate_freq(&sample).unwrap();
        assert_within_tolerance(computed_frequency, 20.0, 1.0);
    }

    #[test]
    fn sanity_non_multiple() {
        let sample = produce_sample(44100, 4000.0, 0.0);
        let yin = Yin::init(0.1, 3000.0, 5000.0, 44100);
        let computed_frequency = yin.estimate_freq(&sample).unwrap();
        let difference = computed_frequency.frequency - 4000.0;
        assert!(difference.abs() < 50.0);
    }

    #[test]
    fn sanity_full_sine() {
        let sample = produce_sample(44100, 443.0, 0.0);
        let yin = Yin::init(0.1, 300.0, 500.0, 44100);
        let computed_frequency = yin.estimate_freq(&sample).unwrap();
        assert_within_tolerance(computed_frequency, 443.0, 1.0);
    }

    #[test]
    fn test_clarity_clean_signal() {
        let sample = produce_sample(44100, 440.0, 0.0); // Clean signal
        let yin = Yin::init(0.1, 300.0, 500.0, 44100);
        let result = yin.estimate_freq(&sample).unwrap();
        assert!(result.clarity > 0.8); // Clean signal should have high clarity
    }

    #[test]
    fn test_clarity_noisy_signal() {
        let sample = produce_sample(44100, 440.0, 0.5); // Noisy signal
        let yin = Yin::init(0.1, 300.0, 500.0, 44100);
        let result = yin.estimate_freq(&sample).unwrap();
        assert!(result.clarity < 0.8); // Noisy signal should have lower clarity
        println!("Noisy signal clarity: {}", result.clarity);
    }

    #[test]
    fn test_clarity_very_noisy_signal() {
        let sample = produce_sample(44100, 440.0, 1.0); // Very noisy signal
        let yin = Yin::init(0.1, 300.0, 500.0, 44100);
        let result = yin.estimate_freq(&sample).unwrap();
        assert!(result.clarity < 0.6); // Very noisy signal should have very low clarity
        println!("Very noisy signal clarity: {}", result.clarity);
    }

    #[test]
    fn readme_doctest() {
        let estimator = Yin::init(0.1, 10.0, 30.0, 80);
        let mut example = vec![];
        let mut prev_value = -1.0;
        for i in 0..80 {
            if i % 2 != 0 {
                example.push(0.0);
            } else {
                prev_value *= -1.0;
                example.push(prev_value);
            }
        }
        let freq = estimator.estimate_freq(&example).unwrap();
        assert_within_tolerance(freq, 20.0, 1.0);
    }
}
