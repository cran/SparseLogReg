/*FindCounts.c the C wrapper for SparseLR */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Error.h>

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "nrutil.h"
#include "my.h"



findcounts(int m_train, int input_dim, int no_of_expts, 
	   double GammaMin, double GammaMax, int no_of_gamma, 
	   int Intkfold, double tol,
	   double **in, int *inclass, int **out)
{
  /* m_train: number of training samples          62 */
  /* input_dim: input dimensions                2000 */
  /* no_of_expts: number of experiments          100 */
  /* GammaMin:                                   .01 */
  /* GammaMax:                                     4 */
  /* no_of_gamma:                                  5 */
  /* Intkfold: Internalkfold                       3 */
  /* tol: tolerance                             1e-6 */
  /* in: input matrix of format m_train * input_dim  */
  /* inclass: input class vector of (1,-1)           */
  /* out: output matrix of format input_dim * 2      */

  long id=-2;
  unsigned long *idx, *idxp, *idxn;
  int i, j, k, k1, k2, k3, itemp;
  int id1, id2;

  int  mtrg;
  int iminarg1, iminarg2;
  int posex_per_fold, negex_per_fold, *trgex_per_fold;
  int  posex_in_last_fold, negex_in_last_fold;
  int mplus, mneg, minex_per_fold, maxex_per_fold;
  int *pos_index, *neg_index, **training_index, **val_index;
  int  *total_y, *y,  **support;
  float *ranpos, *ranneg, *x;
  double valCost, Total_error, gamma, gamma_old;
  double temp, TestError, minError;
  double  **total_input,  **alpha,  *gammaarr;
  double *bias, **xi;
  double *primal_objective_fun, *error_per_fold;
  double *val_output;
  int *CountSV, *tempCountSV, *temptempCountSV;
  
  
  if (Intkfold <= 1) {
    REprintf("For k-fold experiments ensure that k > 1\n");
    exit(1);
  }
  
  gammaarr = dvector(1, no_of_gamma);
  
  if (no_of_gamma == 1)
    gammaarr[1] = GammaMin;
  else {
    temp = (GammaMax-GammaMin)/(no_of_gamma-1);
    for (i=1; i <= no_of_gamma; i++)
      gammaarr[i] = GammaMin+(i-1)*temp;
  }
  
  total_input = dmatrix(1, m_train, 1, input_dim);
  total_y = ivector(1, m_train);
  
  mplus = mneg = 0;
  for (i=1; i <= m_train; i++)  {
    for (j=1; j <= input_dim; j++){
	total_input[i][j] = in[i][j];
	}
    if (inclass[i] == 1) {
      total_y[i] = 1;
      mplus++;
    } else {
      total_y[i] = -1;
      mneg++;
    }
    
  }
      
  CountSV = ivector(1, input_dim);
  tempCountSV = ivector(1, input_dim);
  temptempCountSV = ivector(1, input_dim);
  for (j=1; j <= input_dim; j++)
    CountSV[j] = 0;
    
  val_output = dvector(1, m_train);
  
  pos_index = ivector(1, mplus);
  neg_index = ivector(1, mneg);
  
  /* set the vectors with the indices of positive 
         and negative vectors */
  
  k1 = k2 = 1;
  for (i=1; i <= m_train; i++)  {
    if (total_y[i] == 1) {
      pos_index[k1] = i;
      k1++;
    } else {
      neg_index[k2] = i;
      k2++;
    }
  }
  
  posex_per_fold =  floor(mplus / (double)Intkfold);
  negex_per_fold =  floor(mneg / (double) Intkfold);
  posex_in_last_fold = mplus - (Intkfold-1)*posex_per_fold;
  negex_in_last_fold = mneg - (Intkfold-1)*negex_per_fold;
  
  trgex_per_fold = ivector(1,Intkfold);
  
  minex_per_fold = m_train; maxex_per_fold = 0;
  for (k = 1; k < Intkfold; k++)  {
    trgex_per_fold[k] = (posex_per_fold+negex_per_fold)*(Intkfold-2) + posex_in_last_fold + negex_in_last_fold;
    if (trgex_per_fold[k] > maxex_per_fold)
      maxex_per_fold = trgex_per_fold[k];
    if (trgex_per_fold[k] < minex_per_fold)
      minex_per_fold = trgex_per_fold[k];
  }
  trgex_per_fold[Intkfold] = (posex_per_fold+negex_per_fold)*(Intkfold-1);
  if (trgex_per_fold[Intkfold] > maxex_per_fold)
    maxex_per_fold = trgex_per_fold[Intkfold];
  if (trgex_per_fold[Intkfold] < minex_per_fold)
    minex_per_fold = trgex_per_fold[Intkfold];
  training_index = imatrix(1,Intkfold,1,m_train);
  val_index = imatrix(1,Intkfold,1,m_train-minex_per_fold);
  
  
  alpha = dmatrix(1, Intkfold, 1, input_dim);
  bias = dvector(1, Intkfold);
  xi = dmatrix(1, Intkfold, 1, m_train);
  support = imatrix(1, Intkfold, 1, input_dim);
  primal_objective_fun = dvector(1, Intkfold);
  
  Total_error = 0;
  
  ranpos = vector(1,mplus);
  ranneg = vector(1,mneg);
  
  idxp = lvector(1, mplus);
  idxn = lvector(1, mneg);

  TestError = 0;
  for (k3 = 1; k3 <= no_of_expts; k3++) {
     
    for (i=1; i <= mplus; i++)
      ranpos[i] = ran1(&id);
    for (i=1; i <= mneg; i++)
      ranneg[i] = ran1(&id);
    
    indexx(mplus, ranpos, idxp);
    indexx(mneg, ranneg, idxn);
    
    for (k=1; k <= Intkfold; k++) {
      id1 = id2 = 1;

      /* Set k1 and k2 to the respective pointers in the */
      /* array of positive and negative examples */

      k1 = (k-1)*posex_per_fold+1;
      k2 = (k-1)*negex_per_fold+1;
      if (k < Intkfold) {
	for (i=1; i <= mplus; i++) {
	  if (i >= k1 && i <= k1 + posex_per_fold - 1) {
	    val_index[k][id2] = pos_index[idxp[i]]; id2++;
	  } else {
	    training_index[k][id1] = pos_index[idxp[i]];id1++;
	  }
	}
	for (i=1; i <= mneg; i++) {
	  if (i >= k2 && i <= k2 + negex_per_fold - 1) {
	    val_index[k][id2] = neg_index[idxn[i]]; id2++;
	  } else {
	    training_index[k][id1] = neg_index[idxn[i]];id1++;
	  }
	}
      } else {
	for (i=1; i <= mplus; i++) {
	  if (i >= k1 && i <= k1 + posex_in_last_fold - 1) {
	    val_index[k][id2] = pos_index[idxp[i]]; id2++;
	  } else {
	    training_index[k][id1] = pos_index[idxp[i]];id1++;
	  }
	}
	for (i=1; i <= mneg; i++) {
	  if (i >= k2 && i <= k2 + negex_in_last_fold - 1) {
	    val_index[k][id2] = neg_index[idxn[i]]; id2++;
	  } else {
	    training_index[k][id1] = neg_index[idxn[i]];id1++;
	  }
	}
      }
    }
         
    minError = DBL_MAX;
 
    gamma_old = 0;

    for (k2=1; k2 <= no_of_gamma; k2++) {
                    
      gamma = gammaarr[k2];

      valCost = 0;

      for (j=1; j <= input_dim; j++)
	tempCountSV[j] = 0;

      for (k=1; k <= Intkfold; k++) {

	SparseLOGREGTrain(tol, trgex_per_fold[k], input_dim, total_input, total_y, training_index[k], alpha[k], support[k], gamma_old, gamma, &bias[k], xi[k]);

	valCost += LogregValidate(m_train, trgex_per_fold[k], input_dim, total_input, total_y, training_index[k], val_index[k], alpha[k], support[k], xi[k], &bias[k], gamma, val_output);
                      
	for (j=1; j <= input_dim; j++)
	  if (support[k][j] == 1)
	    tempCountSV[j] += 1;
      }            
      if (minError > valCost/m_train) {
	minError = valCost/m_train;
              
	for (j=1; j <= input_dim; j++)
	  temptempCountSV[j] = tempCountSV[j];
      }

      gamma_old = gamma;
    } /* rof k2 */

    TestError += minError;

    for (j=1; j <= input_dim; j++)
      CountSV[j] += temptempCountSV[j];

  } /* rof k3 */

  itemp = 0;
  x = vector(1, input_dim);
  for (j=1; j <= input_dim; j++) {
    x[j] = CountSV[j];
    if (CountSV[j] > 0)
      itemp++;
  }
  idx = lvector(1, input_dim);
  indexx(input_dim, x, idx);

  for (j=input_dim; j >= 1; j--) {
    i = (int) x[idx[j]];
    if (i > 0) {
      out[j][1] = idx[j];
      out[j][2] = i;
    }
  }
  free_lvector(idx, 1, input_dim);
  free_vector(x, 1, input_dim);
  free_lvector(idxn, 1, mneg);
  free_lvector(idxp, 1, mplus);
  free_vector(ranneg, 1,mneg);
  free_vector(ranpos, 1,mplus);
  free_dvector(primal_objective_fun, 1, Intkfold);
  free_imatrix(support, 1, Intkfold, 1, input_dim);
  free_dmatrix(xi, 1, Intkfold, 1, m_train);
  free_dvector(bias, 1, Intkfold);
  free_dmatrix(alpha, 1, Intkfold, 1, input_dim);
  free_imatrix(training_index,1,Intkfold,1,m_train);
  free_imatrix(val_index, 1, Intkfold, 1, m_train-minex_per_fold);
  free_ivector(trgex_per_fold,1, Intkfold);
  free_ivector(neg_index, 1, mneg);
  free_ivector(pos_index, 1, mplus);
  free_dvector(val_output, 1, m_train);
  free_ivector(tempCountSV, 1, input_dim);
  free_ivector(temptempCountSV, 1, input_dim);
  free_ivector(CountSV,  1, input_dim);
}
