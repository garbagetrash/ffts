use num::complex::Complex32;
use rustfft::*;

use std::f32::consts::PI;
use std::time::Instant;

fn bitrev(i: usize, nbits: u32) -> usize {
    let mut output = 0;
    for b in 0..nbits {
        output <<= 1;
        if (i >> b) & 1 == 1 {
            output ^= 1;
        }
    }
    output
}

struct Radix2Fft {
    nfft: usize,
    bitrevarr: Vec<usize>,
    twiddles: Vec<Complex32>,
}

impl Radix2Fft {
    fn new(nfft: usize) -> Self {
        let nbits = nfft.ilog2();
        let mut br = vec![];
        for i in 0..nfft {
            br.push(bitrev(i, nbits));
        }
        // Precompute twiddle factors
        let warr: Vec<_> = (0..nfft / 2)
            .map(|i| Complex32::new(0.0, -2.0 * PI * i as f32 / nfft as f32).exp())
            .collect();
        Self {
            nfft,
            bitrevarr: br,
            twiddles: warr,
        }
    }

    // Radix-2 DIT:
    // Xk = sum m:(0..N/2 - 1) { x_2m * exp { -2pi*j*(2m)*k / N } } +
    //      sum m:(0..N/2 - 1) { x_2m+1 * exp { -2pi*j*(2m+1)*k / N } }
    // Then factor exp { -2pi*j*k / N } out of right hand side...
    // Xk = sum m:(0..N/2 - 1) { x_2m * exp { -2pi*j*m*k / (N/2) } } +
    //      exp { -2pi*j*k / N } * sum m:(0..N/2 - 1) { x_2m+1 * exp { -2pi*j*m*k / (N/2) } }
    // Let Ek = sum m:(0..N/2 - 1) { x_2m * exp { -2pi*j*m*k / (N/2) } }
    // Let Ok = sum m:(0..N/2 - 1) { x_2m+1 * exp { -2pi*j*m*k / (N/2) } }
    // Xk = Ek + exp { -2pi*j*k / N } * Ok
    // Note that Ek and Ok are both length N/2 DFTs of even and odd indices respectively.
    // Due to periodicity of complex exponential we have:
    // Xk+N/2 = Ek - exp { -2pi*j*k / N } * Ok
    fn execute(&self, _x: &[Complex32]) -> Vec<Complex32> {
        let n = self.nfft;

        // Do bit reversal
        let mut x: Vec<_> = (0..n).map(|i| _x[self.bitrevarr[i]]).collect();

        // Butterfly depth at outer layer, log2(NFFT) layers
        for s in 1..1 + n.ilog2() {
            let m = 2_u32.pow(s) as usize;
            let m2 = m / 2;
            let wstep = n / m;
            // Which butterfly at this layer? Fewer butterflys as we progress.
            for k in (0..n).step_by(m) {
                let mut widx = 0;
                // Each butterfly has a number of arms which increases as we progress.
                for j in 0..m2 {
                    let w = self.twiddles[widx];
                    let tmp = w * x[k + j + m2];
                    x[k + j + m2] = x[k + j] - tmp;
                    x[k + j] += tmp;
                    widx += wstep;
                }
            }
        }
        x
    }
}

// DFT: Xk = sum n:(0..N-1) { xn * exp { -2pi*j*n*k / N } }

fn main() {
    let m = 10000;
    let n = 1024;
    let mut x1: Vec<Complex32> = (0..n).map(|xx| Complex32::new(xx as f32, 0.0)).collect();
    let x2 = x1.clone();
    let mut planner = FftPlanner::new();
    let rfft = planner.plan_fft_forward(n);
    let myfft = Radix2Fft::new(n);

    /*
    // Do it once just to compare answers
    rfft.process(&mut x1);
    x2 = radix2_dit(&x2);
    for i in 0..n {
        assert!((x1[i] - x2[i]).norm() < 1e0);
    }
    */

    let mut bestyet = 9999.0;
    for _ in 0..8 {
        let now = Instant::now();
        for _ in 0..m {
            rfft.process(&mut x1);
        }
        let dt = now.elapsed().as_secs_f64();
        if dt < bestyet {
            bestyet = dt;
        }
    }
    println!(
        "Rust FFT: {} us.",
        1e6 * bestyet / m as f64
    );

    let mut bestyet = 9999.0;
    for _ in 0..8 {
        let now = Instant::now();
        for _ in 0..m {
            myfft.execute(&x2);
        }
        let dt = now.elapsed().as_secs_f64();
        if dt < bestyet {
            bestyet = dt;
        }
    }
    println!(
        "Radix-2 DIT FFT: {} us.",
        1e6 * bestyet / m as f64
    );
}
