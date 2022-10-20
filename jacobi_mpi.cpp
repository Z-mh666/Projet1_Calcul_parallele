//   mpicxx jacobi_mpi.cpp
// Execution:
//   mpirun -np 4 ./a.out 'Nx' 'Ny' 'L'

#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <mpi.h>

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

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int myRank;
  int nbTasks;
  MPI_Comm_size(MPI_COMM_WORLD, &nbTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


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
  double err;  //erreur
  double U0 = 0.5;
  double alpha = 0.5;

  //Boundary conditions + second menbre + solution exacte
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

  // Check time
  double timeInit = MPI_Wtime();

  // Compute local parameters
  int n_start = myRank * (Nx/nbTasks + 1) + 1;
  int n_end = ((myRank+1) * (Nx/nbTasks + 1));
  //n_end = (n_end <= N) ? n_end : N;
  if (n_end > Nx)
  {
    n_end = Nx;
  }
  int Nloc = n_end - n_start + 1;
  cout << "[myRank=" <<  myRank << "] " << n_start << " " << n_end << " " << Nloc << endl;

  //Time loop
  for (int n=0;n<L;n++){
    //MPI_Status status;
    // MPI Exchanges (with botton side)
    MPI_Request reqSendBot, reqRecvBot;
    if(myRank > 0){
      MPI_Isend(&x[n_start*(Ny+2)],   Ny+2, MPI_DOUBLE, myRank-1, 0, MPI_COMM_WORLD, &reqSendBot);
      MPI_Irecv(&x[n_start*(Ny+2)-Ny-2], Ny+2, MPI_DOUBLE, myRank-1, 0, MPI_COMM_WORLD, &reqRecvBot);
    }
    
    // MPI Exchanges (with top side)
    MPI_Request reqSendTop, reqRecvTop;
    if(myRank < nbTasks-1){
      MPI_Isend(&x[n_end*(Ny+2)],  Ny+2, MPI_DOUBLE, myRank+1, 0, MPI_COMM_WORLD, &reqSendTop);
      MPI_Irecv(&x[n_end*(Ny+2)+Ny+2], Ny+2, MPI_DOUBLE, myRank+1, 0, MPI_COMM_WORLD, &reqRecvTop);
    }
    
    // MPI Exchanges (check everything is send/recv)
    if(myRank > 0){
      MPI_Wait(&reqSendBot, MPI_STATUS_IGNORE);
      MPI_Wait(&reqRecvBot, MPI_STATUS_IGNORE);
    }
    if(myRank < nbTasks-1){
      MPI_Wait(&reqSendTop, MPI_STATUS_IGNORE);
      MPI_Wait(&reqRecvTop, MPI_STATUS_IGNORE);
    }
    
    // Spatial loop
    for (int i=n_start;i<=n_end;i++){
      for (int j=1;j<Ny+1;j++){
        sol[j+i*(Ny+2)] = -coef*(h1*(x[j+i*(Ny+2)-1]+x[j+i*(Ny+2)+1])+h2*(
        x[j+i*(Ny+2)-(Nx+2)]+x[j+i*(Ny+2)+Nx+2])-fx[j+i*(Ny+2)]);
      }
    }

    // Switch pointers
    double* tmp;
    tmp = x;
    x = sol;
    sol = tmp;
  }


  // Check time
  MPI_Barrier(MPI_COMM_WORLD);
  if(myRank == 0){
    double timeEnd = MPI_Wtime();
    cout << "Runtime: " << timeEnd-timeInit << endl;
  }

  // Print solution + calcul de l'erreur
  for (int myRankPrint=0; myRankPrint<nbTasks; myRankPrint++){
    if (myRank == myRankPrint){
      ofstream file;
      if(myRank == 0){
        file.open("jacobi_rezu_mpi.dat", ios::out);
      } else {
        file.open("jacobi_rezu_mpi.dat", ios::app);
      }
      
      if(myRank == 0){
        file << '[';
        for(int i=0; i<=Nx+1; i++){
          file << sol[i] << ",";
          err += pow(sol_ex[i]-sol[i],2);
        }
      }
      
      for (int i=n_start; i<=n_end; i++){
        for (int j=0;j<Ny+2;j++){
          file << sol[j+i*(Ny+2)] <<",";
          err += pow(sol_ex[j+i*(Ny+2)]-sol[j+i*(Nx+2)],2);
        }
      }
      
      if(myRank == (nbTasks-1)){
        for (int i=Nx+2;i>1;i--){
          file << sol[(Nx+2)*(Ny+2)-i] << ",";
          err += pow(sol_ex[(Nx+2)*(Ny+2)-i]-sol[(Nx+2)*(Ny+2)-i],2);
        }
        file << sol[(Nx+2)*(Ny+2)-1] << ']' << endl;
      }
      
      file.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  //calcul d'erreur
  if (myRank == 0){
    err = sqrt(err);
    cout << "Absolute error: " << err << endl;
  }

  // Memory deallocation
  delete[] fx;
  delete[] sol;
  delete[] x;
  delete[] sol_ex;

  // Finalize MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}
