extern float ran1(long *);
extern  void indexx(unsigned long n, float arr[], unsigned long indx[]);
extern void SparseLOGREGTrain(double tol, int mtrg, int input_dim, double **input, int *target, int *training_index, double *alpha, int *support,  double gamma_old, double gamma,  double *bias,  double *xi);
extern double LogregValidate(int m, int mtrg, int input_dim, double **input, int *target, int *training_index, int *val_index, double *alpha, int *support, double *xi, double *b, double gamma,  double *val_output);

