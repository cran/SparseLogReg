/*FindGenes.c the C wrapper for SparseLR */

#include <R.h>
#include <Rinternals.h>

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "nrutil.h"
#include "my.h"

findgenes(int m_train, int input_dim, int no_of_expts, 
	  double GammaMin, double GammaMax, int no_of_gamma, 
	  int Intkfold, double tol, int MaxFeatures,
	  double **in, int *inclass, int **infeatures, int **out)
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
  /* inclass: input class vector of (1,-1) */
  /* infeatures: output of findcounts, input_dim*2 */
  /* out: output matrix of format input_dim * 2 */

  long id=-1;
  unsigned long *idx, *idxp, *idxn;
  int i, j, k, k1, k2, k3, k4, k5, itemp;
  int feat_count, minfeatCount, id1, id2;

  unsigned char *input_file, *out_file, *features_file, *junk;
  int iminarg1, iminarg2;
  int posex_per_fold, negex_per_fold, *trgex_per_fold;
  int posex_in_last_fold, negex_in_last_fold;
  int mplus, mneg,  minex_per_fold, maxex_per_fold;
  int *pos_index, *neg_index, **training_index, **val_index;
  int  *total_y,  **support;
  float *ranpos, *ranneg, *x;
  double valCost, Total_error, gamma, gamma_old, FinalGamma;
  double TestError, minError, MinAvgValError, *ErrorAtGamma;
  double temp;
  double  **temp_total_input, **total_input,   **alpha, *gammaarr, *AvgValError;
  double *bias, **xi;
  double *primal_objective_fun, *val_output;
  int rownum = 0;

  junk = cvector(1,80);

  if (Intkfold <= 1) {
    REprintf("For k-fold experiments, ensure that k > 1\n");
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
        
        ErrorAtGamma = dvector(1, no_of_gamma);

      total_input = dmatrix(1, m_train, 1, input_dim);
      temp_total_input = dmatrix(1, m_train, 1, input_dim);
      total_y = ivector(1, m_train);

      mplus= mneg= 0;
      for (i=1; i <= m_train; i++)  {
	for (j=1; j <= input_dim; j++){
	  /* fscanf(fp, "%lf", &total_input[i][j]); */
	  total_input[i][j] = in[i][j];
	}
	if (inclass[j] == 1) {
	  total_y[i] = 1;
	  mplus++;
	} else {
	  total_y[i] = -1;
	  mneg++;
	}

      }


      AvgValError = dvector(1, MaxFeatures);
      for (i=1; i <= MaxFeatures; i++)
          AvgValError[i] = 0;


      x = vector(1, input_dim);

      for (i=1; i <= input_dim; i++)  {

	k = infeatures[i][1];

	x[k] = infeatures[i][2];
      }

      idx = lvector(1, input_dim);
      indexx(input_dim, x, idx);


      feat_count = 0;
      MinAvgValError = DBL_MAX;
      for (k5 = 1; k5 <= MaxFeatures; k5++) {
          val_output = dvector(1, m_train);
 
          for (i=1; i <= m_train; i++)
              for (j=1; j <= input_dim; j++)
                  temp_total_input[i][j] = 0;
 
          for (i=1; i <= no_of_gamma; i++)
              ErrorAtGamma[i] = 0;

          for (k=1; k <= k5; k++) {
               j = idx[input_dim-k5+k];
               out[rownum][k] = (int) idx[input_dim-k5+k];
               for (i=1; i <= m_train; i++)
                    temp_total_input[i][j] = total_input[i][j];
          }
	  rownum++;    
          
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

          if (Intkfold > 1) {
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
          } else {
             trgex_per_fold[1] = m_train;
          }
          training_index = imatrix(1,Intkfold,1,m_train);
          val_index = imatrix(1,Intkfold,1,m_train-minex_per_fold);
          alpha = dmatrix(1, Intkfold, 1, input_dim);
          bias = dvector(1, Intkfold);
          xi = dmatrix(1, Intkfold, 1, m_train);
          support = imatrix(1, Intkfold, 1, input_dim);
          primal_objective_fun = dvector(1, Intkfold);
 

          Total_error = 0;
          /* Store a sequence of random numbers in two arrays */
          ranpos = vector(1,mplus);
          ranneg = vector(1,mneg);

          /* idxp and idxn used for storing the indices of the sorted list */
          idxp = lvector(1, mplus);
          idxn = lvector(1, mneg);
    

          TestError = 0;
          for (k3 = 1; k3 <= no_of_expts; k3++) {

              Rprintf("\n\n######\nExpt %d ::\n", k3);

              for (i=1; i <= mplus; i++)
                   ranpos[i] = ran1(&id);
              for (i=1; i <= mneg; i++)
                   ranneg[i] = ran1(&id);

              indexx(mplus, ranpos, idxp);
              indexx(mneg, ranneg, idxn);

              for (k=1; k <= Intkfold; k++) {
                  id1 = id2 = 1;

                  /***********************************************
                  Set k1 and k2 to the respective pointers in the
                  array of positive and negative examples
                  ***********************************************/

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

                   Rprintf("gamma %g\n", gamma);

                   for (k=1; k <= Intkfold; k++) {

                        SparseLOGREGTrain(tol, trgex_per_fold[k], input_dim, temp_total_input, total_y, training_index[k], alpha[k], support[k], gamma_old, gamma, &bias[k], xi[k]);

                        valCost += LogregValidate(m_train, trgex_per_fold[k], input_dim, temp_total_input, total_y, training_index[k], val_index[k], alpha[k], support[k], xi[k], &bias[k], gamma, val_output);
                      
                   }
                   Rprintf("validation error %g\n--\n", valCost/m_train);
                   
                   ErrorAtGamma[k2] += valCost;
                   
                   if (minError > valCost/m_train) {
                       minError = valCost/m_train;
                   }

                   gamma_old = gamma;
              } /* rof k2 */

              TestError += minError;

          } /* rof k3 */
          temp = DBL_MAX; itemp = 0;
          for (i=1; i <= no_of_gamma; i++)
               if (ErrorAtGamma[i] < temp) {
                   temp = ErrorAtGamma[i]; itemp = i;
               }

          AvgValError[k5] = temp/((mplus+mneg)*no_of_expts);
          if (AvgValError[k5] < MinAvgValError) {
               MinAvgValError = AvgValError[k5];
               minfeatCount = k5;
               FinalGamma = gammaarr[itemp];
          }
          
          if ((k5 > 1 && AvgValError[k5] > AvgValError[k5-1]) || (AvgValError[k5] > MinAvgValError)) {
              feat_count += 1;
              if (feat_count == 4)  {
                  k = minfeatCount;
                  for (i=1; i <= m_train; i++)
                     for (j=1; j <= input_dim; j++)
                       temp_total_input[i][j] = 0;
                  for (k2=1; k2 <= k; k2++) {
                     j = idx[input_dim-k+k2];
                     out[rownum][k2] = (int) idx[input_dim-k+k2];
                     for (i=1; i <= m_train; i++)
                          temp_total_input[i][j] = total_input[i][j];
		  }
		  rownum++;

                 for (i=1; i <= m_train; i++)
                    training_index[1][i] = i;

               /*****************************************************
                 Now train using the entire training set
               *****************************************************/
              SparseLOGREGTrain(tol, m_train, input_dim, temp_total_input, total_y, training_index[1], alpha[1], support[1], 0, FinalGamma, &bias[1], xi[1]);

   
              Rprintf("\n\nFinal Model at gamma = %g\n", FinalGamma);
              for (i=1; i <= input_dim; i++) 
                  if (support[1][i] != 0) {
                     Rprintf("%d %g\n", i, alpha[1][i]);
                  }
              Rprintf("Bias %g\n", -bias[1]);
              k5 = MaxFeatures;
            }
          } else
                 feat_count = 0;
          free_lvector(idxn, 1, mneg);
          free_lvector(idxp, 1, mplus);
          free_vector(ranneg, 1,mneg);
          free_vector(ranpos, 1,mplus);
          free_dvector(primal_objective_fun, 1, Intkfold);
          free_imatrix(support, 1, Intkfold, 1, input_dim);
          free_dmatrix(xi, 1, Intkfold, 1, m_train);
          free_dvector(bias, 1, Intkfold);
          free_dmatrix(alpha, 1, Intkfold, 1, input_dim);
          free_imatrix(val_index, 1, Intkfold, 1, m_train-minex_per_fold);
          free_imatrix(training_index,1,Intkfold,1,m_train);
          free_ivector(trgex_per_fold,1, Intkfold);
          free_ivector(neg_index, 1, mneg);
          free_ivector(pos_index, 1, mplus);
          free_dvector(val_output, 1, m_train);
      } /* rof k5 */
        free_lvector(idx,1,input_dim);
        free_vector(x,1, input_dim);
        free_dvector(AvgValError, 1, MaxFeatures);
        free_ivector(total_y, 1, m_train);
        free_dmatrix(temp_total_input, 1, m_train, 1, input_dim);
        free_dmatrix(total_input, 1, m_train, 1, input_dim);
        free_dvector(ErrorAtGamma, 1, no_of_gamma);
        free_dvector(gammaarr, 1,  no_of_gamma);
}
