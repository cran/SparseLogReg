/*logreg.c the C wrapper for SparseLR */

#include <R.h>
#include <Rinternals.h>


#include <stdio.h>
#include <math.h>
#include <float.h>
#include "nrutil.h"

#define DMAX(a,b) (dmaxarg1=(a),dmaxarg2=(b), (dmaxarg1) > (dmaxarg2) ? (dmaxarg1) : (dmaxarg2))
#define DMIN(a,b) (dminarg1=(a),dminarg2=(b), (dminarg1) < (dminarg2) ? (dminarg1) : (dminarg2))
#define imin(a,b) (iminarg1=(a),iminarg2=(b), (iminarg1) < (iminarg2) ? (iminarg1) : (iminarg2))

extern double *dvector();
extern void free_dvector();
extern double fabs(), exp(), log();

double ComputePrimal(int mtrg, int input_dim, double gamma, int *support, double *alpha, double *xi)
{
   int i, j;
   double temp = 0;
   for (i=1; i <= mtrg; i++) 
      temp +=  log(1.0+exp(xi[i]));
   for (j=1; j <= input_dim; j++)
      if (support[j] == 1)
         temp += gamma*fabs(alpha[j]);
   return(temp);
}

void ComputeFAndDelta(int j, int mtrg,  double *xi, int *training_index, int *target, double **input, double *F, double *Delta)
{
   int i;
   double temp1, temp2 = 0, temp = 0;

   if (j == 0) {
      for (i=1; i <= mtrg; i++) {
          temp1 = exp(xi[i]);
          temp += temp1*target[training_index[i]]/(1+temp1);
          temp2 += temp1/((1+temp1)*(1+temp1));
      }
   } else {
      for (i=1; i <= mtrg; i++) {
          temp1 = exp(xi[i]);
          temp += temp1*target[training_index[i]]*input[training_index[i]][j]/(1+temp1);
          temp2 += temp1*input[training_index[i]][j]*input[training_index[i]][j]/((1+temp1)*(1+temp1));
      }
   }
   *F = temp;
   *Delta = temp2;
}

void UpdateXi(int j, int mtrg,  double *xi, int *training_index, int *target, double var_old, double var_new, double **input)
{
   int i;

   if (j == 0) {
       for (i=1; i <= mtrg; i++) 
           xi[i] += target[training_index[i]]*(var_new-var_old);

   } else {
       for (i=1; i <= mtrg; i++) 
           xi[i] += target[training_index[i]]*(var_old-var_new)*input[training_index[i]][j];
  }
}

void InitializeVariables(int mtrg, int input_dim, int *support,  double *alpha, double *xi,  double *bias)
{	
	int i,j;

        for (i=1; i <= mtrg; i++) 
	      xi[i]  = 0; 
        for (j=1; j <= input_dim; j++)
              support[j] = alpha[j] = 0;

        *bias = 0;
}

int FindMaxViolator(int k, int mtrg, int input_dim, double *xi, int *training_index, int *target, int *support, double **input, double *alpha, double gamma, double *F, double *Delta, double tol)
{

   int i, j, v;
   double MaxViol, viol_j, F_t, Delta_t;
   double dmaxarg1, dmaxarg2, dminarg1, dminarg2;


   if (k) {
       v = 0;
       ComputeFAndDelta(0, mtrg, xi, training_index, target, input,  &F_t, &Delta_t);
       MaxViol = fabs(F_t);

       for (j=1; j <= input_dim; j++) {
            if (support[j] == 1) {
                  ComputeFAndDelta(j, mtrg, xi, training_index, target, input,  &F_t, &Delta_t);
                  if (alpha[j] > 0)
                      viol_j =  fabs(gamma-F_t);
                  else
                      viol_j = fabs(gamma+F_t);

                  if (viol_j > MaxViol) {
                       MaxViol = viol_j; v = j;
                       *F = F_t; *Delta = Delta_t;
                  } 
            }
       }
    } else {
      v = -1;
      MaxViol = tol;
      for (j=1; j <= input_dim; j++) {
            if (support[j] == 0) {
                  ComputeFAndDelta(j, mtrg, xi, training_index, target, input,  &F_t, &Delta_t);
                    viol_j = DMAX(F_t-gamma, -F_t-gamma);
                    if (viol_j < 0)
                         viol_j = 0;
                    if (viol_j > MaxViol) {
                       MaxViol = viol_j; v = j;
                       *F = F_t; *Delta = Delta_t;
                    }
           }
       }
    }

    if (MaxViol > tol)
        return (v);
    else 
        return(-1);
}
     

void OptimizeForAlpha(int v, int mtrg, int *support, int *training_index, int *target, double *alpha, double *bias, double *xi, double **input, double *F, double *Delta,  double gamma, double tol, int input_dim)
{
    int i, flag;
    double temp, alpha_cur, alpha_new, slope_v, slope_0;
    double  Fv_at_0, Deltav_at_0, L, H;
    double *temp_xi;

    L = H = flag = 0;

    if (v == 0) {

       L = -DBL_MAX; H = DBL_MAX;

       slope_v = *F;

       alpha_cur = *bias;

    } else if (support[v] != 0) {
        
        temp_xi = dvector(1, mtrg);

        for (i=1; i <= mtrg; i++)
            temp_xi[i] = xi[i];

        alpha_cur = alpha[v]; alpha_new = 0;

        UpdateXi(v, mtrg, temp_xi, training_index, target, alpha_cur, alpha_new, input);
        ComputeFAndDelta(v, mtrg, temp_xi, training_index, target, input, &Fv_at_0, &Deltav_at_0);

        if ((alpha_cur > 0 && (gamma-*F) > 0 && (gamma-Fv_at_0) > 0 ) || (alpha_cur < 0 && (-gamma-*F) < 0 && (-gamma-Fv_at_0) < 0 )) {
            
            flag = 1;
            for (i=1; i <= mtrg; i++)
                xi[i] = temp_xi[i];
 
            *F = Fv_at_0; *Delta = Deltav_at_0;
            support[v] = alpha[v]  = 0;

            if (alpha_cur > 0) {
                slope_v = -gamma-Fv_at_0;
                L = -DBL_MAX; H = 0;
            } else {
                slope_v = gamma-Fv_at_0;
                L = 0; H = DBL_MAX;
            }

            alpha_cur = alpha_new = 0;
            if (-gamma-*F < 0 && gamma-*F > 0)
                slope_v = 0;
        } else {

            if (alpha[v] > 0) {

                slope_v = gamma-*F;

                if  (gamma-*F > 0 && gamma-Fv_at_0 < 0) {

                    L = 0; H = alpha[v];

                } else if (gamma-*F < 0) {

                    L = alpha[v]; H = DBL_MAX;
                }

            } else {

                slope_v = -gamma-*F;

               if (-gamma-*F < 0 && -gamma-Fv_at_0 > 0) {

                    L = alpha[v]; H = 0;

               } else if (-gamma-*F > 0) {

                    L = -DBL_MAX; H = alpha[v];

               }
            }
        }

        free_dvector(temp_xi, 1, mtrg);        

    } else {
       
         alpha_cur = alpha[v];
   
        if (gamma-*F < 0) {

             L = 0; H = DBL_MAX;

             slope_v = gamma-*F;
     
        } else if (-gamma-*F > 0) {

             L = -DBL_MAX; H = 0;
             
             slope_v = -gamma-*F;

        }

    }

    while (fabs(slope_v) > .1*tol) {

        alpha_new = alpha_cur - slope_v/(*Delta);

        if (alpha_new <= L || alpha_new >= H)
         
             alpha_new = (L+H)/2.0;
        
        UpdateXi(v, mtrg,  xi, training_index, target, alpha_cur, alpha_new, input);
        ComputeFAndDelta(v, mtrg,  xi, training_index, target, input, F, Delta);

        if (v != 0) {
        
           if (alpha_new > 0)

              slope_v = gamma-*F;
           else
   
             slope_v = -gamma-*F;

        } else

           slope_v = *F;
   
        if (slope_v > .1*tol)
 
             H = alpha_new;

        else if (slope_v < -.1*tol)

             L = alpha_new;

        alpha_cur = alpha_new;

     };


  

        
    if (v > 0) {
       alpha[v] = alpha_new;
       if (fabs(alpha[v]) > 0) 
           support[v] = 1;
    } else
       *bias = alpha_new;

}


void SparseLOGREGTrain(double tol, int mtrg, int input_dim, double **input, int *target, int *training_index, double *alpha, int *support,  double gamma_old, double gamma,  double *bias,  double *xi)
{
	int count, flag, i, j, k, id, p_maxviol;
	double primal, temp, MaxViol, Delta_v;
         double dmaxarg1, dmaxarg2, dminarg1, dminarg2;
        double F, Delta;

        if (gamma_old < 1e-6) 
            InitializeVariables(mtrg, input_dim, support, alpha, xi,  bias);

        id = FindMaxViolator(0, mtrg, input_dim, xi, training_index, target, support, input, alpha, gamma, &F, &Delta, tol);
        if (id == -1)
           id = FindMaxViolator(1, mtrg, input_dim, xi, training_index, target,support,  input, alpha, gamma, &F, &Delta, tol);
        while ( id >= 0) {
           do {
              OptimizeForAlpha(id, mtrg, support, training_index, target, alpha, bias, xi, input, &F, &Delta,  gamma, tol, input_dim);
              id = FindMaxViolator(1, mtrg, input_dim, xi, training_index, target, support, input, alpha, gamma, &F, &Delta, tol);
           } while (id >= 0);
          id = FindMaxViolator(0, mtrg, input_dim, xi, training_index, target, support, input, alpha, gamma, &F, &Delta, tol);
        };
}




double LogregValidate(int m, int mtrg, int input_dim, double **input, int *target, int *training_index, int *val_index, double *alpha, int *support, double *xi, double *b, double gamma,  double *val_output)
{
	int i, j;
	static int k = 0;
	double temp, cost;
	
	cost = 0;
	for (i=1; i <= m-mtrg; i++) {
		k++;
		temp = 0;
		for (j=1; j <= input_dim; j++)
			temp += alpha[j]*input[val_index[i]][j];
                temp -= *b;

                val_output[i] = temp;
		
                if (target[val_index[i]] > 0) {
                    if (temp < 0)
                        cost += 1.0; 
                } else {
                    if (temp > 0)
                        cost += 1.0;
                }
	}
	return(cost);
}

