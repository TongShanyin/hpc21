// g++ -std=c++11 -g val_test01_solved.cpp -o val_test01_solved
// valgrind ./val_test01_solved

# include <cstdlib>
# include <iostream>

using namespace std;

int main ( );
void f ( int n );

//****************************************************************************80

int main ( )

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for TEST01.
//
//  Discussion:
//
//    TEST01 calls F, which has a memory "leak".  This memory leak can be
//    detected by VALGRID.
//
//    There are 3 errors detected by valgrind.
//     (1) Error: Mismatched free() / delete / delete []
//         Since we use malloc() before, we need to use free() other than delete() to free memory;
//     (2)&(3): we use malloc() for length n but the loop for i<=n, which requires n+1, exceeds the length.
//
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    18 May 2011
//
{
  int n = 10;

  cout << "\n";
  cout << "TEST01\n";
  cout << "  C++ version.\n";
  cout << "  A sample code for analysis by VALGRIND.\n";

  f ( n );
//
//  Terminate.
//
  cout << "\n";
  cout << "TEST01\n";
  cout << "  Normal end of execution.\n";

  return 0;
}
//****************************************************************************80

void f ( int n )

//****************************************************************************80
//
//  Purpose:
//
//    F computes N+1 entries of the Fibonacci sequence.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    18 May 2011
//
{
  int i;
  int *x;

  x = ( int * ) malloc ( n * sizeof ( int ) );

  x[0] = 1;
  cout << "  " << 0 << "  " << x[0] << "\n";

  x[1] = 1;
  cout << "  " << 1 << "  " << x[1] << "\n";

  for ( i = 2; i <n; i++ )
  {
    x[i] = x[i-1] + x[i-2]; // Invalid read of size 4
    cout << "  " << i << "  " << x[i] << "\n";
  }

  // delete [] x; // Error: Mismatched free() / delete / delete []
  free(x); // Since we use malloc() before, we need to use free() other than delete() to free memory

  return;
}
