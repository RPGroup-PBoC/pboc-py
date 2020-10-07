functions {
    // Function to generate posterior predictive samples
    vector gp_ppc_rng(
        real[] t_ppc,   // time points where to evaluate ppc
        vector y,       // OD measurements
        real[] t_data,  // time points for OD measurements
        real alpha,     // marginal standard deviation
        real rho,       // time scale
        real sigma,     // measurement error
        real delta      // extra value added for numerical stability
    ) {
    // Define variables for evaluation
    int N_data = rows(y);       // number of data points
    int N_ppc = size(t_ppc);    // number of time points on ppc
    vector[N_ppc] f_ppc;        // posterior predictive samples
    {
        // Compute conditional mean <f(x*) | f(x), x, y>
        // 0. Generate covariance matrix for the data
        // 0.1 Compute exponentiated quadratic kernel
        matrix[N_data, N_data] K_exp = cov_exp_quad(t_data, alpha, rho);

        // 0.2 Add observation error to diagonal
        matrix[N_data, N_data] Kxx = K_exp 
        + diag_matrix(rep_vector(square(sigma), N_data));

        // 1. Perform Cholesky decomposition Kxx = Lxx Lxx'
        matrix[N_data, N_data] Lxx = cholesky_decompose(Kxx);

        // 2. Solve for b = inv(Lxx) y. Since L_K is a triangular matrix we can 
        // use the mdivide_left_tri_low function
        vector[N_data] b = mdivide_left_tri_low(Lxx, y);

        // 3. Solve a = inv(Lxx') inv(Lxx) y. Since Lxx' is a triangular 
        // matrix we can use the mdivide_right_tri_low function
        vector[N_data] a = mdivide_right_tri_low(
            b', Lxx
        )';

        // 4. Multiply by covariance matrix between f(x*) and f(x)
        // 4.1 Generate covariance matrix for both data an ppc Kxx*
        matrix[N_data, N_ppc] Kxx_star= cov_exp_quad(
            t_data, t_ppc, alpha, rho
        );
        // 4.2 Multiply to obtain conditional mean <f(x*)|f(x),x,y>
        vector[N_ppc] fx_star_cond = (Kxx_star' * a);

        // Compute conditional covariance cov(f(x*), f(x*))| f(x), x, y

        // 1. Evaluate v = inv(Lxx) * Kxx*
        matrix[N_data, N_ppc] v = mdivide_left_tri_low(
            Lxx, Kxx_star
        );
        // 2. Evaluate conditional covariance matrix
        // 2.1 Compute exponentiated quadratic kernel for f(x*)
        matrix[N_ppc, N_ppc] Kx_star_x_star= cov_exp_quad(t_ppc, alpha, rho) ;

        // 2.2 Evaluate Kx*|x = Kx*x* - v' * v with small numerical value
        matrix[N_ppc, N_ppc] Kx_star_cond = Kx_star_x_star- v' * v
                                      + diag_matrix(rep_vector(delta, N_ppc));

        // Generate random samples given the variance and covariance functions
        // for the ppc samples
        f_ppc = multi_normal_rng(fx_star_cond, Kx_star_cond);
    }
    return f_ppc;
    }
}

data {
    // Data from OD measurements
    int<lower=1> N;     // number of data points
    real t[N];          // time points where measurements were taken
    vector[N] y;        // optical density measurements

    // Posterior Predictive Checks
    int<lower=1> N_predict;     // number of points where to evalute ppc
    real t_predict[N_predict];  // time points where to evaluate ppc
}

parameters {
    real<lower=0> rho;      // time scale
    real<lower=0> alpha;    // marginal standard deviation
    real<lower=0> sigma;    // measurement standard deviation
}

model {
    // Define covariance matrix k(t, t')
    matrix[N, N] cov_exp =  cov_exp_quad(t, alpha, rho);
    matrix[N, N] cov = cov_exp + diag_matrix(rep_vector(square(sigma), N));
    // Perform a Cholesky decomposition of the matrix, this means rewrite the
    // covariance matrix cov = L_cov L_cov'
    matrix[N, N] L_cov = cholesky_decompose(cov);
    
    // Sample data from a multinomial Gaussian with mean zero and rather than
    // covariance matrix, a Cholesky decomposed matrix
    y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
}

generated quantities {
    // Generate posterior predictive samples for the Gaussian process
    vector[N_predict] f_predict = gp_ppc_rng(
        t_predict, y, t, alpha, rho, sigma, 1e-10
    );
    // Generate posterior predictive samples for the observation process
    vector[N_predict] y_predict;
    for (n in 1:N_predict) {
        y_predict[n] = normal_rng(f_predict[n], sigma);
    }
}