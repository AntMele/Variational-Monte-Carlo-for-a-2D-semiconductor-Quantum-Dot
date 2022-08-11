#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multimin.h>


int N = 5; // number of electrons
double omega = 1.0;
double a [] = {2.0, 2.0/3.0}; // Jastrow coefficient (antiparallel, parallel)
double b [] = {1.10, 0.45}; // Jastrow coefficient
int * spins; // vector containing the spin projection along z of each electron: +1 or -1


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
    switch(N){
    case 2:
        x = sqrt(gsl_pow_2(R[0]-R[2]) + gsl_pow_2(R[1]-R[3]));
        return exp(-0.5*omega*(gsl_pow_2(R[0])+gsl_pow_2(R[1])+gsl_pow_2(R[2])+gsl_pow_2(R[3])) + a[0]*x/(1+b[0]*x));
    case 3:
        for(int i = 1; i < 3; i++){
            for(int j = 0; j < i; j++){
                s = spins[i] * spins[j] == 1;
                x = sqrt(gsl_pow_2(R[2*i]-R[2*j]) + gsl_pow_2(R[2*i+1]-R[2*j+1]));
                out += a[s]*x/(1+b[s]*x);
            }
        }
        x = 0;
        for(int i = 0; i < 6; i++){
            x += gsl_pow_2(R[i]);
        }
        return fabs((R[4] - R[2] + R[5] - R[3]) * exp(-0.5*omega*x + out));
    case 5:
        for(int i = 1; i < 5; i++){
            for(int j = 0; j < i; j++){
                s = spins[i] * spins[j] == 1;
                x = sqrt(gsl_pow_2(R[2*i]-R[2*j]) + gsl_pow_2(R[2*i+1]-R[2*j+1]));
                out += a[s]*x/(1+b[s]*x);
            }
        }
        x = 0;
        for(int i = 0; i < 10; i++){
            x += gsl_pow_2(R[i]);
        }
        return fabs((R[8] - R[6] + R[9] - R[7]) * (R[2]*R[5] - R[3]*R[4] + R[0]*(R[3]-R[5]) + R[1]*(R[4]-R[2])) * exp(-0.5*omega*x + out));
    case 6:
        for(int i = 1; i < N; i++){
            for(int j = 0; j < i; j++){
                s = spins[i] * spins[j] == 1;
                x = sqrt(gsl_pow_2(R[2*i]-R[2*j]) + gsl_pow_2(R[2*i+1]-R[2*j+1]));
                out += a[s]*x/(1+b[s]*x);
            }
        }
        out = exp(out);
        gsl_matrix * A = gsl_matrix_alloc(3,3);
        gsl_permutation * p = gsl_permutation_alloc(3);
        x = 0;
        for(int i = 0; i < 3; i++){
            x += gsl_pow_2(R[2*i]) + gsl_pow_2(R[2*i+1]);
            gsl_matrix_set(A, 0, i, 1.0);
            gsl_matrix_set(A, 1, i, R[2*i]);
            gsl_matrix_set(A, 2, i, R[2*i+1]);
        }
        gsl_linalg_LU_decomp(A, p, &s);
        out *= gsl_linalg_LU_det(A, s) * exp(-0.5 * omega * x);
        x = 0;
        for(int i = 0; i < 3; i++){
            x += gsl_pow_2(R[2*i+6]) + gsl_pow_2(R[2*i+7]);
            gsl_matrix_set(A, 0, i, 1.0);
            gsl_matrix_set(A, 1, i, R[2*i+6]);
            gsl_matrix_set(A, 2, i, R[2*i+7]);
        }
        gsl_linalg_LU_decomp(A, p, &s);
        out *= gsl_linalg_LU_det(A, s) * exp(-0.5 * omega * x);
        gsl_matrix_free(A);
        gsl_permutation_free(p);
        return fabs(out);
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

// I compute the local energy given the electron positions R
double localEnergy(double R[]){
    double E = 0;
    double gradJ [2*N];
    for(int i = 0; i < 2*N; i++){
        gradJ[i] = 0;
    }
    double laplJ = 0;

    double rij;
    int s;
    double x1, x2, x3;
    for(int i = 1; i < N; i++){
        for(int j = 0; j < i; j++){
            s = spins[i] * spins[j] == 1;
            rij = sqrt(gsl_pow_2(R[2*i]-R[2*j]) + gsl_pow_2(R[2*i+1]-R[2*j+1]));
            x1 = 1 / rij;
            x2 = 1 / gsl_pow_2(1 + b[s]*rij);
            x3 = a[s] * x1 * x2;
            E += x1;
            x1 = x3 * x2 * (1 - gsl_pow_2(b[s]*rij));
            x2 = (R[2*i] - R[2*j]) * x3;
            x3 = (R[2*i+1] - R[2*j+1]) * x3;
            laplJ += x1;
            gradJ[2*i] += x2;
            gradJ[2*j] -= x2;
            gradJ[2*i+1] += x3;
            gradJ[2*j+1] -= x3;
        }
    }
    laplJ *= 2;

    double gradMF [2*N];
    switch(N){
    case 2:
        E += 2 * omega;
        for(int i = 0; i < 4; i++){
            gradMF[i] = -omega * R[i];
        }
        break;
    case 3:
        E += 4 * omega;
        for(int i = 0; i < 6; i++){
            gradMF[i] = -omega * R[i];
        }
        x1 = 1 / (R[4] - R[2] + R[5] - R[3]);
        gradMF[2] -= x1;
        gradMF[3] -= x1;
        gradMF[4] += x1;
        gradMF[5] += x1;
        break;
    case 5:
        E += 8 * omega;
        for(int i = 0; i < 10; i++){
            gradMF[i] = -omega * R[i];
        }
        x1 = 1 / (R[8] - R[6] + R[9] - R[7]);
        gradMF[6] -= x1;
        gradMF[7] -= x1;
        gradMF[8] += x1;
        gradMF[9] += x1;
        x1 = 1 / (R[2]*R[5] - R[3]*R[4] + R[0]*(R[3]-R[5]) + R[1]*(R[4]-R[2]));
        gradMF[0] += (R[3] - R[5]) * x1;
        gradMF[1] += (R[4] - R[2]) * x1;
        gradMF[2] += (R[5] - R[1]) * x1;
        gradMF[3] += (R[0] - R[4]) * x1;
        gradMF[4] += (R[1] - R[3]) * x1;
        gradMF[5] += (R[2] - R[0]) * x1;
        break;
    case 6:
        E += 10 * omega;
        gsl_matrix * A = gsl_matrix_alloc(3,3);
        gsl_matrix * dxA = gsl_matrix_alloc(3,3);
        gsl_matrix * dyA = gsl_matrix_alloc(3,3);
        gsl_permutation * p = gsl_permutation_alloc(3);
        for(int i = 0; i < 3; i++){
            x1 = exp(-0.5 * omega * (gsl_pow_2(R[2*i]) + gsl_pow_2(R[2*i+1])));
            gsl_matrix_set(A, 0, i, x1 * M_SQRT1_2 / sqrt(omega));
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
        for(int i = 0; i < 3; i++){
            x1 = exp(-0.5 * omega * (gsl_pow_2(R[2*i+6]) + gsl_pow_2(R[2*i+7])));
            gsl_matrix_set(A, 0, i, x1 * M_SQRT1_2 / sqrt(omega));
            gsl_matrix_set(A, 1, i, R[2*i+6] * x1);
            gsl_matrix_set(A, 2, i, R[2*i+7] * x1);
            gsl_matrix_set(dxA, 0, i, -omega * R[2*i+6] * gsl_matrix_get(A, 0, i));
            gsl_matrix_set(dxA, 1, i, (1 - omega * gsl_pow_2(R[2*i+6])) * x1);
            gsl_matrix_set(dxA, 2, i, -omega * R[2*i+6] * gsl_matrix_get(A, 2, i));
            gsl_matrix_set(dyA, 0, i, -omega * R[2*i+7] * gsl_matrix_get(A, 0, i));
            gsl_matrix_set(dyA, 1, i, -omega * R[2*i+7] * gsl_matrix_get(A, 2, i));
            gsl_matrix_set(dyA, 2, i, (1 - omega * gsl_pow_2(R[2*i+7])) * x1);
        }
        gsl_linalg_LU_decomp(A, p, &s);
        gsl_linalg_LU_invx(A, p);
        for(int i = 0; i < 3; i++){
            gradMF[2*i+6] = 0;
            gradMF[2*i+7] = 0;
            for(int j = 0; j < 3; j++){
                gradMF[2*i+6] += gsl_matrix_get(A, i, j) * gsl_matrix_get(dxA, j, i);
                gradMF[2*i+7] += gsl_matrix_get(A, i, j) * gsl_matrix_get(dyA, j, i);
            }
        }
        gsl_matrix_free(A);
        gsl_matrix_free(dxA);
        gsl_matrix_free(dyA);
        gsl_permutation_free(p);
        break;
    default:
        printf("LOL\n");
        exit(EXIT_FAILURE);
    }

    x1 = 0;
    for(int i = 0; i < 2*N; i++){
        x1 += gradJ[i] * gradMF[i];
        laplJ += gsl_pow_2(gradJ[i]);
    }
    return E - (0.5*laplJ + x1);
}


int main(){
    srand(0);//time(NULL)
    spins = malloc(sizeof(int)*N);
    switch(N){
    case 2:
        spins[0] = 1;
        spins[1] = -1;
        break;
    case 3:
        spins[0] = 1;
        spins[1] = -1;
        spins[2] = -1;
        break;
    case 5:
        spins[0] = 1;
        spins[1] = 1;
        spins[2] = 1;
        spins[3] = -1;
        spins[4] = -1;
        break;
    case 6:
        spins[0] = 1;
        spins[1] = 1;
        spins[2] = 1;
        spins[3] = -1;
        spins[4] = -1;
        spins[5] = -1;
        break;
    default:
        printf("W.I.P.\n");
        exit(EXIT_FAILURE);
    }

    double R [2*N];
    init(R);
    double psiR = wavefunction(R);

    //printf("%f\n", psiR);
    //printf("%f\n", localEnergy(R));



/*
    // I see everything
    int n;
    printf("    %+8.1f  %+4.1f", psiR, localEnergy(R));
    for(int j = 0; j < 2*N; j++){
        printf("  %+4.2f",R[j]);
    }printf("\n");
    for(int i = 0; i < 100; i++){
        n = metropolis(R, &psiR, 4.0);
        printf("%2d  %+8.1f  %+4.1f", n, psiR, localEnergy(R));
        for(int j = 0; j < 2*N; j++){
            printf("  %+4.2f",R[j]);
        }printf("\n");
    }
*/




    // I estimate the total energy of the system
    int numSteps = 5000000;
    int numTrash = 1000;
    int numBlock = 1;
    double delta = 4.0;
    FILE * file = fopen("E.txt", "w");
    double E;
    double avg = 0;
    double avg2 = 0;
    int n = 0;
    for(int i = 0; i < numTrash; i++){
        n += metropolis(R, &psiR, delta);
    }
    printf("Acceptance ratio for the first %d trashed steps: %f \n", numTrash, n/(2.0*N*numTrash));
    n = 0;
    for(int i = 0; i < numSteps / numBlock; i++){
        for(int j = 0; j < numBlock; j++){
            n += metropolis(R, &psiR, delta);
        }
        E = localEnergy(R);
        avg += E;
        avg2 += E*E;
        fprintf(file, "%.15f\n", E);
    }
    printf("Acceptance ratio for the other %d steps: %f \n", numSteps, n/(2.0*N*numSteps));
    avg /= numSteps / numBlock;
    avg2 /= numSteps / numBlock;
    printf("E = %.15f +- %.15f\n", avg, sqrt((avg2 - gsl_pow_2(avg))/(numSteps/numBlock - 1)));
    fclose(file);




/*
    // Correlation as function of Delta and tau
    int numSteps = 500000;
    int numTrash = 1000;
    int numCorr = 100;
    double * E = malloc(sizeof(double)*(numSteps + 1));
    FILE * file = fopen("corr.txt", "w");
    for(int k = 0; k < 20; k++){
        double delta = 0.5 * (k + 1);
        printf("Delta = %f \n", delta);
        int n = 0;
        for(int i = 0; i < numTrash; i++){
            n += metropolis(R, &psiR, delta);
        }
        printf("Acceptance ratio for the first %d trashed steps: %f \n", numTrash, n/(2.0*N*numTrash));
        n = 0;
        E[0] = localEnergy(R);
        for(int i = 0; i < numSteps; i++){
            n += metropolis(R, &psiR, delta);
            E[i + 1] = localEnergy(R);
        }
        printf("Acceptance ratio for the other %d steps: %f \n\n", numSteps, n/(2.0*N*numSteps));
        double corr[numCorr];
        double avg = mean(E, numSteps+1);
        double std = sigma(E, avg, numSteps+1) * sqrt(numSteps);
        for(int i = 0; i < numCorr; i++){
            corr[i] = 0;
            for(int j = 0; j < numSteps + 1 - i; j++){
                corr[i] += E[j] * E[j + i];
            }
            corr[i] = (corr[i]/(numSteps + 1 - i) - gsl_pow_2(avg)) / gsl_pow_2(std);
            fprintf(file, "%.15f;", corr[i]);
        }
        fprintf(file, "\n");
        //srand(0);
        init(R);
        psiR = wavefunction(R);
    }
    fclose(file);
*/



/*
    // Acceptance ratio as function of Delta
    int n;
    for(int j = 0; j < 200; j++){
        double delta = j / 10.0;
        //double delta = 0.004 * gsl_pow_uint(1.0718913192, j);
        for(int i = 0; i < 1000; i++){
            metropolis(R, &psiR, delta);
        }
        n = 0;
        for(int i = 0; i < 10000; i++){
            n += metropolis(R, &psiR, delta);
        }
        printf("%.15f\n", n/(10000.0*2*N));
        srand(0);
        init(R);
        psiR = wavefunction(R);
    }
*/

    return 0;
}
