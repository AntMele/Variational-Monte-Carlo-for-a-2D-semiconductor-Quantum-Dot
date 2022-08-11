#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multimin.h>

int wave=0; //wave=0: L=0 S=1, wave=1: L=2 S=0, wave=2: L=0 S=0
int N = 4; // number of electrons
double omega = 1.0;
double err[3];
double a [] = {2.0, 2.0/3.0}; // Jastrow coefficient (antiparallel, parallel)
double b [] = {0.5, 2}; // Jastrow coefficient
int * spins; // vector containing the spin projection along z of each electron: +1 or -1
double bf[3][2];

// I compute the mean value of the vector vec of size n
double mean(double vet [], unsigned int n){
    double sum = 0;
    for(int i = 0; i < n; i++){
        sum += vet[i];
    }
    return sum / n;
}

// I compute the standard deviation of the mean value avg of the vector vec of size n
// Notice in the end I divide by sqrt(n-1)
double sigma(double vet [], double avg, unsigned int n){
    double avg2 = 0;
    for(int i = 0; i < n; i++){
        avg2 += gsl_pow_2(vet[i]);
    }
    avg2 /= n;
    return sqrt((avg2 - gsl_pow_2(avg)) / (n - 1));
}

// Initialize the position vector displacing the N particle on a circumference of radius r which is determined minimizing the potential energy
void init(double R[]){
    double r = 0;
    for(int i = 1; i < N; i++){
        r += 1 / sin(i*M_PI/N);
    }
    r = pow(r/gsl_pow_2(2*omega), 1.0/3.0);
    for(int i = 0; i < N; i++){
        R[2*i] = r * cos(i*2*M_PI/N);
        R[2*i+1] = r * sin(i*2*M_PI/N);
    }
}

// Returns the absolute value of the wave function given the electron positions R
double wavefunction(double R[]){
    double x;
    int s;
    double out = 0;
    for(int i = 1; i < N; i++){
        for(int j = 0; j < i; j++){
            s = spins[i] * spins[j] == 1;
            x = sqrt(gsl_pow_2(R[2*i]-R[2*j]) + gsl_pow_2(R[2*i+1]-R[2*j+1]));
            out += a[s]*x/(1+b[s]*x);
        }
    }
    out = exp(out);
    
    x=0;
    for(int i = 0; i < N; i++){
        x += gsl_pow_2(R[2*i]) + gsl_pow_2(R[2*i+1]);
    }

    switch(wave){ //wave=0: L=0 S=1, wave=1: L=2 S=0, wave=2: L=0 S=0
    case 0:
            ;
            gsl_matrix * A = gsl_matrix_alloc(3,3);
            gsl_permutation * p = gsl_permutation_alloc(3);
            for(int i = 0; i < 3; i++){
                gsl_matrix_set(A, 0, i, 1.0);
                gsl_matrix_set(A, 1, i, R[2*i]);
                gsl_matrix_set(A, 2, i, R[2*i+1]);
            }
            gsl_linalg_LU_decomp(A, p, &s);
            out *= gsl_linalg_LU_det(A, s) * exp(-0.5 * omega * x);
            
            gsl_matrix_free(A);
            gsl_permutation_free(p);
            return fabs(out);

    case 1:
        return out*fabs( (R[2] - R[0])*(R[6] - R[4]) + (R[3] - R[1])*(R[7] - R[5]) )* exp(-0.5*omega*x);
    case 2:
        return out*fabs( (R[2] - R[0])*(R[7] - R[5]) + (R[3] - R[1])*(R[6] - R[4]) )* exp(-0.5*omega*x) ;
    default:
        printf("don't mess with me\n");
        exit(EXIT_FAILURE);
    }
}

// I make a Metropolis step of size delta for each coordinate of R. I return the number of accepted steps.
int metropolis(double R[], double *psiR, double delta){
    double propR [2*N];
    for(int i = 0; i < 2*N; i++){
        propR[i] = R[i];
    }
    double propPsiR;

    int count = 0;
    for(int i = 0; i < 2*N; i++){
        propR[i] += delta * (((double)rand())/RAND_MAX - 0.5);
        propPsiR = wavefunction(propR);
        if(propPsiR >= *psiR || ((double)rand())/RAND_MAX < gsl_pow_2(propPsiR / *psiR)){
            R[i] = propR[i];
            *psiR = propPsiR;
            count++;
        }else{
            propR[i] = R[i];
        }
    }
    return count;
}

// I compute the logarithmic gradient of the mean field wave function
void computeGradMF(double R[], double gradMF[]){
    int s;
    double x1;
    switch(wave){ //wave=0: L=0 S=1, wave=1: L=2 S=0, wave=2: L=0 S=0
    case 0:
            ;
            gsl_matrix * A = gsl_matrix_alloc(3,3);
            gsl_matrix * dxA = gsl_matrix_alloc(3,3);
            gsl_matrix * dyA = gsl_matrix_alloc(3,3);
            gsl_permutation * p = gsl_permutation_alloc(3);
            for(int i = 0; i < 3; i++){
                x1 = exp(-0.5 * omega * (gsl_pow_2(R[2*i]) + gsl_pow_2(R[2*i+1])));
                gsl_matrix_set(A, 0, i, x1 );
                gsl_matrix_set(A, 1, i, R[2*i] * x1);
                gsl_matrix_set(A, 2, i, R[2*i+1] * x1);
                gsl_matrix_set(dxA, 0, i, -omega * R[2*i] * gsl_matrix_get(A, 0, i));
                gsl_matrix_set(dxA, 1, i, (1 - omega * gsl_pow_2(R[2*i])) * x1);
                gsl_matrix_set(dxA, 2, i, -omega * R[2*i] * gsl_matrix_get(A, 2, i));
                gsl_matrix_set(dyA, 0, i, -omega * R[2*i+1] * gsl_matrix_get(A, 0, i));
                gsl_matrix_set(dyA, 1, i, -omega * R[2*i+1] * gsl_matrix_get(A, 2, i));
                gsl_matrix_set(dyA, 2, i, (1 - omega * gsl_pow_2(R[2*i+1])) * x1);
            }
            gsl_linalg_LU_decomp(A, p, &s);
            gsl_linalg_LU_invx(A, p);
            for(int i = 0; i < 3; i++){
                gradMF[2*i] = 0;
                gradMF[2*i+1] = 0;
                for(int j = 0; j < 3; j++){
                    gradMF[2*i] += gsl_matrix_get(A, i, j) * gsl_matrix_get(dxA, j, i);
                    gradMF[2*i+1] += gsl_matrix_get(A, i, j) * gsl_matrix_get(dyA, j, i);
                }
            }
            gradMF[6] = -omega * R[6];
            gradMF[7] = -omega * R[7];
            gsl_matrix_free(A);
            gsl_matrix_free(dxA);
            gsl_matrix_free(dyA);
            gsl_permutation_free(p);
            break;
        case 1:
        for(int i = 0; i < 2*N; i++){
            gradMF[i] = -omega * R[i];
        }
        x1 = 1 / ( (R[2] - R[0])*(R[6] - R[4]) + (R[3] - R[1])*(R[7] - R[5]) );
        gradMF[0] += -(R[6] - R[4])*x1;
        gradMF[1] += -(R[7] - R[5])*x1;
        gradMF[2] += (R[6] - R[4])*x1;
        gradMF[3] += (R[7] - R[5])*x1;
        gradMF[4] += -(R[2] - R[0])*x1;
        gradMF[5] += -(R[3] - R[1])*x1;
        gradMF[6] += (R[2] - R[0])*x1;
        gradMF[7] += (R[3] - R[1])*x1;
        break;
    case 2:
        for(int i = 0; i < 2*N; i++){
            gradMF[i] = -omega * R[i];
        }
        x1 = 1 / ( (R[2] - R[0])*(R[7] - R[5]) + (R[3] - R[1])*(R[6] - R[4]) );
        gradMF[0] += -(R[7] - R[5])*x1;
        gradMF[1] += -(R[6] - R[4])*x1;
        gradMF[2] += (R[7] - R[5])*x1;
        gradMF[3] += (R[6] - R[4])*x1;
        gradMF[4] += -(R[3] - R[1])*x1;
        gradMF[5] += -(R[2] - R[0])*x1;
        gradMF[6] += (R[3] - R[1])*x1;
        gradMF[7] += (R[2] - R[0])*x1;
        break;
    default:
        printf("LOL\n");
        exit(EXIT_FAILURE);
    }
}

void energyFunctional(const gsl_vector *v, void *params, double *f, gsl_vector *df){
    b[0] = gsl_vector_get(v, 0);
    b[1] = gsl_vector_get(v, 1);
    double R [2*N];
    init(R);
    double psiR = wavefunction(R);
    int numSteps = 10000;
    int numTrash = 1000;
    int numBlock = 1;
    double delta = 4.0;
    for(int i = 0; i < numTrash; i++){
        metropolis(R, &psiR, delta);
    }
    int n = 0;
    double gradMF [2*N];
    double E;
    double avg = 0;
    double avg2 = 0;
    double gradJ [2*N];
    double laplJ;
    double Edx;
    double avgdx = 0;
    double gradJdx [2*N];
    double laplJdx;
    double Edy;
    double avgdy = 0;
    double gradJdy [2*N];
    double laplJdy;
    double wdx;
    double wdy;
    double sumWdx = 0;
    double sumWdy = 0;
    double db = 5e-2; // approximated differential: E'(b) = (E(b+db) - E(b)) / db
    double rkl;
    int s;
    double x1, x2, x3, x2d;
    for(int i = 0; i < numSteps / numBlock; i++){
        for(int j = 0; j < numBlock; j++){
            n += metropolis(R, &psiR, delta);
        }
        computeGradMF(R, gradMF);

        E = 0;
        Edx = 0;
        Edy = 0;
        for(int k = 0; k < 2*N; k++){
            gradJ[k] = 0;
            gradJdx[k] = 0;
            gradJdy[k] = 0;
        }
        laplJ = 0;
        laplJdx = 0;
        laplJdy = 0;
        wdx = 0;
        wdy = 0;

        for(int k = 1; k < N; k++){
            for(int l = 0; l < k; l++){
                s = spins[k] * spins[l] == 1;
                rkl = sqrt(gsl_pow_2(R[2*k]-R[2*l]) + gsl_pow_2(R[2*k+1]-R[2*l+1]));
                x1 = 1 / rkl;
                x2 = 1 / gsl_pow_2(1 + b[s]*rkl);
                x3 = a[s] * x1 * x2;
                E += x1;
                Edx += x1;
                Edy += x1;
                x1 = x3 * x2 * (1 - gsl_pow_2(b[s]*rkl));
                x2 = (R[2*k] - R[2*l]) * x3;
                x3 = (R[2*k+1] - R[2*l+1]) * x3;
                laplJ += x1;
                gradJ[2*k] += x2;
                gradJ[2*l] -= x2;
                gradJ[2*k+1] += x3;
                gradJ[2*l+1] -= x3;
                x2d = 1 / gsl_pow_2(1 + (b[s]+db)*rkl);
                if(s){
                    laplJdx += x1;
                    gradJdx[2*k] += x2;
                    gradJdx[2*l] -= x2;
                    gradJdx[2*k+1] += x3;
                    gradJdx[2*l+1] -= x3;
                    x3 = a[s] * x2d / rkl;
                    laplJdy += x3 * x2d * (1 - gsl_pow_2((b[s]+db)*rkl));
                    x2 = (R[2*k] - R[2*l]) * x3;
                    x3 = (R[2*k+1] - R[2*l+1]) * x3;
                    gradJdy[2*k] += x2;
                    gradJdy[2*l] -= x2;
                    gradJdy[2*k+1] += x3;
                    gradJdy[2*l+1] -= x3;
                    wdy += a[s]*rkl*(1/(1+(b[s]+db)*rkl) - 1/(1+b[s]*rkl));
                }else{
                    laplJdy += x1;
                    gradJdy[2*k] += x2;
                    gradJdy[2*l] -= x2;
                    gradJdy[2*k+1] += x3;
                    gradJdy[2*l+1] -= x3;
                    x3 = a[s] * x2d / rkl;
                    laplJdx += x3 * x2d * (1 - gsl_pow_2((b[s]+db)*rkl));
                    x2 = (R[2*k] - R[2*l]) * x3;
                    x3 = (R[2*k+1] - R[2*l+1]) * x3;
                    gradJdx[2*k] += x2;
                    gradJdx[2*l] -= x2;
                    gradJdx[2*k+1] += x3;
                    gradJdx[2*l+1] -= x3;
                    wdx += a[s]*rkl*(1/(1+(b[s]+db)*rkl) - 1/(1+b[s]*rkl));
                }
            }
        }
        laplJ *= 2;
        laplJdx *= 2;
        laplJdy *= 2;
        wdx = exp(2 * wdx);
        wdy = exp(2 * wdy);

        x1 = 0;
        x2 = 0;
        x3 = 0;
        for(int k = 0; k < 2*N; k++){
            x1 += gradJ[k] * gradMF[k];
            x2 += gradJdx[k] * gradMF[k];
            x3 += gradJdy[k] * gradMF[k];
            laplJ += gsl_pow_2(gradJ[k]);
            laplJdx += gsl_pow_2(gradJdx[k]);
            laplJdy += gsl_pow_2(gradJdy[k]);
        }
        E -= 0.5*laplJ + x1;
        Edx -= 0.5*laplJdx + x2;
        Edy -= 0.5*laplJdy + x3;

        avg += E;
        avg2 += E*E;
        avgdx += wdx * Edx;
        avgdy += wdy * Edy;
        sumWdx += wdx;
        sumWdy += wdy;
    }
    avg /= numSteps / numBlock;
    avg2 /= numSteps / numBlock;
    avgdx /= sumWdx;
    avgdy /= sumWdy;

    *f = avg;
    gsl_vector_set(df, 0, (avgdx-avg)/db);
    gsl_vector_set(df, 1, (avgdy-avg)/db);

    E = 6 * omega;
    err[wave]=sqrt((avg2 - gsl_pow_2(avg)) / (numSteps/numBlock - 1));
    printf("b = (%.12f, %.12f)   E = %.12f +- %.12f   acc. ratio %f\n",
           b[0],
           b[1],
           avg + E,
           err[wave],
           n / (2.0 * N * numSteps));
    printf("%f %f %f %f\n", avgdx+E, avgdx-avg, avgdy+E, avgdy-avg);
    bf[wave][0]=b[0];
    bf[wave][1]=b[1];
}



double energy(){
    printf("WAVE: %d _______________________________________________________________\n", wave);
    double en=0;
    srand(0);//time(NULL)
    spins = malloc(sizeof(int)*N);
    switch(wave){
    case 0:
        spins[0] = 1;
        spins[1] = 1;
        spins[2] = 1;
        spins[3] = -1;
        break;
    case 1:
        spins[0] = 1;
        spins[1] = 1;
        spins[2] = -1;
        spins[3] = -1;
        break;
    case 2:
        spins[0] = 1;
        spins[1] = 1;
        spins[2] = -1;
        spins[3] = -1;
        break;
    default:
        printf("W.I.P.\n");
        exit(EXIT_FAILURE);
    }


    // I minimize the energy functional
    gsl_multimin_function_fdf func;
    func.n = 2;
    func.fdf = &energyFunctional;

    gsl_vector * ss = gsl_vector_alloc(2); // initial parameters
    gsl_vector_set(ss, 0, b[0]);
    gsl_vector_set(ss, 1, b[1]);

    const gsl_multimin_fdfminimizer_type * T = gsl_multimin_fdfminimizer_steepest_descent;
    gsl_multimin_fdfminimizer * s = gsl_multimin_fdfminimizer_alloc(T, 2);
    gsl_multimin_fdfminimizer_set(s, &func, ss, 0.2, 0.2); // initial step and scaling factor of the step
    int iter = 0;
    int status;
    do{
        iter++;
        status = gsl_multimin_fdfminimizer_iterate(s);
        if(status){
            printf("Status %d \n", status);
            en=s->f + 6*omega;
            break;
        }
        status = gsl_multimin_test_gradient(s->gradient, 1e-3); // stopping criteria: gradient smaller than 1e-3
        if(status == GSL_SUCCESS){
            printf ("Minimum found at:\n");
            printf ("%5d %.12f %.12f %.12f\n", iter, gsl_vector_get(s->x, 0), gsl_vector_get(s->x, 1), s->f);
        }
    }while(status == GSL_CONTINUE && iter < 100);
    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(ss);
    wave++;
    return en;
}

int main()
{
    b[0] = 0.85;
    b[1] = 0.8;
    double E0=energy(); // L=0 S=1
    b[0] = 0.81;
    b[1] = 0.8;
    double E1=energy(); // L=2 S=0
    b[0] = 0.93;
    b[1] = 0.8;
    double E2=energy(); // L=0 S=0
    printf("Energy L=0 S=1: %.12f +- %.12f, b=(%.12f, %.12f)\n", E0, err[0],bf[0][0],bf[0][1]);
    printf("Energy L=2 S=0: %.12f +- %.12f, b=(%.12f, %.12f) \n", E1, err[1], bf[1][0],bf[1][1]);
    printf("Energy L=0 S=0: %.12f +- %.12f, b=(%.12f, %.12f)\n", E2, err[2],bf[2][0],bf[2][1]);
    return 0;
}

