//   g++ gauss_seidel.cpp
// Execution:
//   ./a.out 'Nx' 'Ny' 'L'

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <time.h> 

using namespace std;

double a = 1.;
double b = 1.;

//fonction V
double V(double y){
  return 1-cos(2*M_PI*y/b);
}

//solution exacte
double u(double x,double y,double U0,double alpha){
  return U0*(1+alpha*(1-sin(M_PI/2*x))*(1-cos(2*M_PI*y)));
}

//second membre
double f(double x,double y,double U0,double alpha){
  return 4*M_PI*M_PI*alpha*U0*(1-sin(M_PI/2*x))*cos(2*M_PI*y)-M_PI*M_PI/4*alpha*U0*(1-cos(2*M_PI*y))*sin(M_PI/2*x);
}


int main(int argc, char* argv[]){
    // Problem parameters
  if (argc!=4){
    cout << "You need to input 3 variables: Nx, Ny and L, here there are " << argc-1 << " variables" << endl;
    return 1;
  }
  int Nx = atoi(argv[1]);
  int Ny = atoi(argv[2]);
  int L = atoi(argv[3]);
  
  double dx = a/(Nx+1.);
  double dy = b/(Ny+1.);
  double h1 = 1./dx/dx;
  double h2 = 1./dy/dy;
  
  // Memory allocation + Initial solution
  double* fx = new double[(Nx+2)*(Ny+2)];  //second membre
  double* sol = new double[(Nx+2)*(Ny+2)];  //solution approchee
  double* sol_ex = new double[(Nx+2)*(Ny+2)];  //solution exacte
  double* x = new double[(Nx+2)*(Ny+2)];  //vecteur initial
  double coef = 1./(-2.*h1-2.*h2);  //coefficients diagonaux
  double err = 0;  //erreur
  double sol_norm = 0;  //norme de la solution
  double err_rel; //erreur relatif
  double U0 = 0.5;
  double alpha = 0.5;

  //Boundary conditions
  for (int i=0;i<Nx+2;i++){
    for (int j=0;j<Ny+2;j++){
      if((j==0)||(j==Ny+1)||(i==Nx+1)){
        x[i+j*(Nx+2)] = U0;
        sol[i+j*(Nx+2)] = U0;
        sol_ex[i+j*(Nx+2)] = U0;
      }
      else if (i==0){
        x[i+j*(Nx+2)] = U0*(1+alpha*V(j*dy));
        sol[i+j*(Nx+2)] = U0*(1+alpha*V(j*dy));
        sol_ex[i+j*(Nx+2)] = U0*(1+alpha*V(j*dy));
      }
      else{
        x[i+j*(Nx+2)] = 1.;
        sol_ex[i+j*(Nx+2)] = u(i*dx,j*dy,U0,alpha);
      }
      fx[i+j*(Nx+2)] = f(i*dx,j*dy,U0,alpha);
    }
  }

  //check time
  clock_t start,end;
  start = clock();

  //Time loop
  for (int n=0;n<L;n++){
    // Spatial loop

    //red points
    for (int i=1;i<Nx+1;i++){
      for (int j=1;j<Ny+1;j++){
        if (((i%2==1)&&(j%2==1)) || ((i%2==0)&&(j%2==0))){
          sol[j+i*(Ny+2)] = -coef*(h1*(x[j+i*(Ny+2)-1]+x[j+i*(Ny+2)+1])+h2*(
          x[j+(i-1)*(Ny+2)]+x[j+(i+1)*(Ny+2)])-fx[j+i*(Ny+2)]);
        }
      }
    }

    //black points
    for (int i=1;i<Nx+1;i++){
      for (int j=1;j<Ny+1;j++){
        if (((i%2==0)&&(j%2==1)) || ((i%2==1)&&(j%2==0))){
          sol[j+i*(Ny+2)] = -coef*(h1*(x[j+i*(Ny+2)-1]+x[j+i*(Ny+2)+1])+h2*(
          x[j+(i-1)*(Ny+2)]+x[j+(i+1)*(Ny+2)])-fx[j+i*(Ny+2)]);
        }
      }
    }

    // Switch pointers
    double* tmp;
    tmp = x;
    x = sol;
    sol = tmp;
  }

  //check time
  end = clock();  

  // Print solution
  ofstream file;
  file.open("gauss_rezu.dat");
  file << '[';
  for (int n=0; n<(Nx+2)*(Ny+2)-1; n++){
    file << sol[n] << ',';
  }
  file << sol[(Nx+2)*(Ny+2)-1] << ']' << endl;

  //calcul d'erreur
  for (int i=0;i<(Nx+2)*(Ny+2);i++){
    err += (sol_ex[i]-sol[i])*(sol_ex[i]-sol[i]);
    sol_norm += sol_ex[i]*sol_ex[i];
  }
  err = sqrt(err);
  sol_norm = sqrt(sol_norm);
  err_rel = err/sol_norm;

  cout << "Runtime : "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
  cout << "Absolute error: " << err << endl;

  // Memory deallocation
  delete[] fx;
  delete[] sol;
  delete[] x;
  delete[] sol_ex;

  return 0;
}
