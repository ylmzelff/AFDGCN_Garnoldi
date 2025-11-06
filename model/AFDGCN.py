# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 12:36:50 2024

@author: Administrator
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import GRAPH, ALGO


from typing import Optional, Tuple
import math
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from scipy.special import legendre

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Linear, Parameter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.linalg import expm
import scipy.io as sio

from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs, eigsh
from torch_geometric.utils import is_torch_sparse_tensor, spmm, to_edge_index


from scipy.special import gamma, factorial

#! /usr/bin/env python
#
def imtqlx ( n, d, e, z ):

#*****************************************************************************80
#
## IMTQLX diagonalizes a symmetric tridiagonal matrix.
#
#  Discussion:
#
#    This routine is a slightly modified version of the EISPACK routine to
#    perform the implicit QL algorithm on a symmetric tridiagonal matrix.
#
#    The authors thank the authors of EISPACK for permission to use this
#    routine.
#
#    It has been modified to produce the product Q' * Z, where Z is an input
#    vector and Q is the orthogonal matrix diagonalizing the input matrix.
#    The changes consist (essentially) of applying the orthogonal 
#    transformations directly to Z as they are generated.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    15 June 2015
#
#  Author:
#
#    John Burkardt.
#
#  Reference:
#
#    Sylvan Elhay, Jaroslav Kautsky,
#    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of
#    Interpolatory Quadrature,
#    ACM Transactions on Mathematical Software,
#    Volume 13, Number 4, December 1987, pages 399-415.
#
#    Roger Martin, James Wilkinson,
#    The Implicit QL Algorithm,
#    Numerische Mathematik,
#    Volume 12, Number 5, December 1968, pages 377-383.
#
#  Parameters:
#
#    Input, integer N, the order of the matrix.
#
#    Input, real D(N), the diagonal entries of the matrix.
#
#    Input, real E(N), the subdiagonal entries of the
#    matrix, in entries E(1) through E(N-1). 
#
#    Input, real Z(N), a vector to be operated on.
#
#    Output, real LAM(N), the diagonal entries of the diagonalized matrix.
#
#    Output, real QTZ(N), the value of Q' * Z, where Q is the matrix that 
#    diagonalizes the input symmetric tridiagonal matrix.
#
  import numpy as np
  #from r8_epsilon import r8_epsilon
  
  from sys import exit

  lam = np.zeros ( n )
  for i in range ( 0, n ):
    lam[i] = d[i]

  qtz = np.zeros ( n )
  for i in range ( 0, n ):
    qtz[i] = z[i]

  if ( n == 1 ):
    return lam, qtz

  itn = 30

  prec = 2.220446049250313E-016

  e[n-1] = 0.0

  for l in range ( 1, n + 1 ):

    j = 0

    while ( True ):

      for m in range ( l, n + 1 ):

        if ( m == n ):
          break

        if ( abs ( e[m-1] ) <= prec * ( abs ( lam[m-1] ) + abs ( lam[m] ) ) ):
          break

      p = lam[l-1]

      if ( m == l ):
        break

      if ( itn <= j ):
        print ( '' )
        print ( 'IMTQLX - Fatal error!' )
        print ( '  Iteration limit exceeded.' )
        exit ( 'IMTQLX - Fatal error!' )

      j = j + 1
      g = ( lam[l] - p ) / ( 2.0 * e[l-1] )
      r = np.sqrt ( g * g + 1.0 )

      if ( g < 0.0 ):
        t = g - r
      else:
        t = g + r

      g = lam[m-1] - p + e[l-1] / ( g + t )
 
      s = 1.0
      c = 1.0
      p = 0.0
      mml = m - l

      for ii in range ( 1, mml + 1 ):

        i = m - ii
        f = s * e[i-1]
        b = c * e[i-1]

        if ( abs ( g ) <= abs ( f ) ):
          c = g / f
          r = np.sqrt ( c * c + 1.0 )
          e[i] = f * r
          s = 1.0 / r
          c = c * s
        else:
          s = f / g
          r = np.sqrt ( s * s + 1.0 )
          e[i] = g * r
          c = 1.0 / r
          s = s * c

        g = lam[i] - p
        r = ( lam[i-1] - g ) * s + 2.0 * c * b
        p = s * r
        lam[i] = g + p
        g = c * r - b
        f = qtz[i]
        qtz[i]   = s * qtz[i-1] + c * f
        qtz[i-1] = c * qtz[i-1] - s * f

      lam[l-1] = lam[l-1] - p
      e[l-1] = g
      e[m-1] = 0.0

  for ii in range ( 2, n + 1 ):

     i = ii - 1
     k = i
     p = lam[i-1]

     for j in range ( ii, n + 1 ):

       if ( lam[j-1] < p ):
         k = j
         p = lam[j-1]

     if ( k != i ):

       lam[k-1] = lam[i-1]
       lam[i-1] = p

       p        = qtz[i-1]
       qtz[i-1] = qtz[k-1]
       qtz[k-1] = p

  return lam, qtz


#! /usr/bin/env python
#
def p_polynomial_zeros ( nt ):

#*****************************************************************************80
#
## P_POLYNOMIAL_ZEROS: zeros of Legendre function P(n,x).
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    16 March 2016
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer NT, the order of the rule.
#
#    Output, real T(NT), the zeros.
#


  a = np.zeros ( nt )

  b = np.zeros ( nt )

  for i in range ( 0, nt ):
    ip1 = i + 1
    b[i] = ip1 / np.sqrt ( 4 * ip1 * ip1 - 1 )

  c = np.zeros ( nt )
  c[0] = np.sqrt ( 2.0 )

  t, w = imtqlx ( nt, a, b, c )

  return  t+1# 0.9*t#for [0, 2] interval


def j_polynomial_zeros ( nt, alpha, beta):

#*****************************************************************************80
#
## P_POLYNOMIAL_ZEROS: zeros of Legendre function P(n,x).
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    19 October 2023
#
#  Author:
#
#    Dr. Mustafa Coşkun
#
#  Parameters:
#
#    Input, integer NT, the order of the rule, upper and lower are bounds.
#
#    Output, real T(NT), the zeros.
#

  ab = alpha + beta
  abi = 2.0 + ab
  # define the zero-th moment
  zemu = (np.power(2.0,(ab + 1.0))*gamma(alpha + 1.0)*gamma(beta + 1.0))/gamma(abi)
  
  
  x = np.zeros(nt)
  bj = np.zeros(nt)
  
  
  x[0] = (beta -alpha)/abi
  bj[0] = np.sqrt(4.0*(1.0 + alpha)*(1.0 + beta)/((abi + 1.0)*abi*abi))
  a2b2 = beta*beta-alpha*alpha

  for i in range ( 2, nt +1 ):
    abi = 2.0*i + ab
    x[i-1] = a2b2 / ((abi -2.0)*abi)
    abi = np.power(abi,2)
    bj[i-1] = np.sqrt((4.0*i*(i+alpha)*(i + beta)* (i + ab))/((abi -1.0)*abi))

 #bjs = np.sqrt(bj)
  c = np.zeros ( nt )
  c[0] = np.sqrt ( zemu )

  t, w = imtqlx ( nt, x, bj, c )

  return t+1 # for 0.9*t#[0, 2] interval

def g_fullRWR(x):
    return (1)/(1 - x) 
    #return x/(1-x) - x**2 


def g_0(x):
    return (0.1)/(1 - x) 
def g_1(x):
    return (1)/(1 - x)
def g_2(x):
    return ((x)/(1 - x))
def g_3(x):
    return (x**2)/(1 - x)   
def g_4(x):
    return (1)/(1 + 25*x**2)


def g_par(x):
    return 1/(1 + x)

def g_appRWR(x,Ksteps):
    sum = 0
    for k in range(Ksteps):
        sum = sum + x**k
    return 0.1*sum

def g_heat(x,Ksteps):
    t = 5
    sum = 0
    for k in range(Ksteps):
        sum = sum + (t**k)/np.math.factorial(k)
    return np.math.exp(-sum)
def g_band_rejection(x):
    return (1-np.exp(-10*(x-1)**2))

def g_band_pass(x):
    return np.exp(-10*(x-1)**2)

def g_low_pass(x):
    return np.exp(-10*x**2)
def g_high_pass(x):
    return 1-np.exp(-10*x**2)

def g_comb(x):
    return np.abs(np.sin(np.pi*x))
    
def filter_jackson(c):
    N = len(c)
    n = np.arange(N)
    tau = np.pi/(N+1)
    g = ((N-n+1)*np.cos(tau*n) + np.sin(tau*n)/np.tan(tau))/(N+1)
    c = np.multiply(g,c)
    return c

# def filter_jackson(c):
# 	"""
# 	Apply the Jackson filter to a sequence of Chebyshev	moments. The moments 
# 	should be arranged column by column.

# 	Args:
# 		c: Unfiltered Chebyshev moments

# 	Output:
# 		cf: Jackson filtered Chebyshev moments
# 	"""

# 	N = len(c)
# 	n = np.arange(N)
# 	tau = np.pi/(N+1)
# 	g = ((N-n+1)*np.cos(tau*n)+np.sin(tau*n)/np.tan(tau))/(N+1)
# 	g.shape = (N,1)
# 	c = g*c 
#     #print(c)
    
# 	return c

def g_Ours(x):
    sum = 1*1 + 1*x + 4*x**2 + 5*x**3
    return sum
def runge(x):
  """In some places the x range is expanded and the formula give as 1/(1+x^2)
  """
  return 1 / (1 +  x ** 2)

def polyfitA(x,y,n):

    m = x.size
    Q = np.ones((m, 1), dtype=object)
    H = np.zeros((n+1, n), dtype=object)
    k = 0
    j = 0
    for k in range(n):
        q = np.multiply(x,Q[:,k])
        #print(q)
        for j in range(k):
            H[j,k] = np.dot(Q[:,j].T,(q/m))
            q = q - np.dot(H[j,k],(Q[:,j]))
        H[k+1,k] = np.linalg.norm(q)/np.sqrt(m)
        Q = np.column_stack((Q, q/H[k+1,k]))
    #print(Q)
    #print(Q.shape)
    d = np.linalg.solve(Q.astype(np.float64), y.astype(np.float64))
    return d, H

def polyvalA(d,H,s):
    inputtype = H.dtype.type
    M = len(s)
    W = np.ones((M,1), dtype=inputtype)
    n = H.shape[1]
    #print("Complete H", H)
    k = 0
    j = 0
    for k in range(n):
        w = np.multiply(s,W[:,k])
        for j in range(k):
            #print( "H[j,k]",H[j,k])
            w = w -np.dot(H[j,k],( W[:,j]))
        W = np.column_stack((W, w/H[k+1,k]))
    y = W @ d
    return y, W


def t_polynomial_zeros(x0, x1, n):
  return (x1 - x0) * (np.cos((2*np.arange(1, n + 1) - 1)/(2*n)*np.pi) + 1) / 2  + x0

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

def s_polynomial_zeros(n):
    temp = Parameter(torch.Tensor(n+1))
    temp.data.fill_(1.0)
    coe_tmp=F.relu(temp)
    coe=coe_tmp.clone()
    for i in range(n):
        coe[i]=coe_tmp[0]*cheby(i,math.cos((n+0.5)*math.pi/(n+1)))
        for j in range(1,n+1):
            x_j=math.cos((n-j+0.5)*math.pi/(n+1))
            coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
        coe[i]=2*coe[i]/(n+1)
    return coe

def poylfitA_Cheby(x,y,n,a,b):
    omega = (b-a)/2
    rho = -((b+a)/(b-a))
    
    IN = np.identity(n+1)
    X = np.diag(x)
    
    firstElement = (2/omega)*X + 2*rho*IN
    firstRow = np.concatenate((firstElement, -IN), axis=1)
    secondRow = np.concatenate((IN, np.zeros((n+1, n+1), dtype=object)), axis=1)
    Xcurly = np.concatenate((firstRow, secondRow), axis=0)
    
   
    m = x.size
    
    T = np.ones((m, 1), dtype=object)

    #This is just convert (n,) to (n,1) shape
    x = x.reshape(-1,1)
    Q = np.concatenate((x,T),axis = 0)

  
    H = np.zeros((n+1, n), dtype=object)
    k = 0
    j = 0
    for k in range(n):
        q = np.matmul(Xcurly,Q[:,k])
        #print(q)
        for j in range(k):
            H[j,k] = np.dot(Q[:,j].T,(q/m))
            q = q - np.dot(H[j,k],(Q[:,j]))
        H[k+1,k] = np.linalg.norm(q)/np.sqrt(m)
        Q = np.column_stack((Q, q/H[k+1,k]))
  
    newQ = Q[n+1:2*(n+1),:];

    d = np.linalg.solve(newQ.astype(np.float64), y.astype(np.float64))
    return d, H

def compare_fitA(f, x, Vander, Threeterm,x0, x1):
  y = f(x)
  n = x.size-1

  if(Vander):
      coefficients = Vandermonde(x, y)

  else:
      if(Threeterm):
          coefficients, H = poylfitA_Cheby(x,y,n,x0,x1)    
      else:
          coefficients, H = polyfitA(x, y, n)
  #K = coefficients.shape[0]
  # for k in range(K-1, -1, -1):
  #     print(coefficients[k], k)
  #print(coefficients)
  return coefficients

def m_polynomial_zeros (x0, x1, n):
    return  np.linspace(x0, x1,n)
def compare_fit_panelA(f, polyname, Vandermonde, Threeterm,degree, x0, x1,zoom=False):
   # Male equedistance
   #x = np.linspace(x0, x1,10)
   if (polyname == 'Monomial'):
       x = m_polynomial_zeros(x0, x1, degree)
   elif (polyname == 'Chebyshev'):    
       x = t_polynomial_zeros(x0, x1, degree)
   elif (polyname == 'Legendre'):
       x = p_polynomial_zeros(degree)
   elif (polyname == 'Jacobi'):    
       x = j_polynomial_zeros(degree,0,1)
   else:
    print ('Give proper polynomial to interpolate\n')
    print ('Calling Monimal as default\n')
    x = m_polynomial_zeros(x0, x1, degree)
    
   return compare_fitA(f, x, Vandermonde,Threeterm, x0,x1)
def Vandermonde(x, y):
  """Return a polynomial fit of order n+1 to n points"""
  #z = np.polyfit(x, y, x.size + 1)
 
  V = np.vander(x) # Vandermonde matrix
  coeffs = np.linalg.solve(V, y) # f_nodes must be a column vector
  return coeffs

class ARNOLDI(MessagePassing):
    r"""The approximate personalized propagation of neural predictions layer
    from the `"Predict then Propagate: Graph Neural Networks meet Personalized
    PageRank" <https://arxiv.org/abs/1810.05997>`_ paper

    .. math::
        \mathbf{X}^{(0)} &= \mathbf{X}

        \mathbf{X}^{(k)} &= (1 - \alpha) \mathbf{\hat{D}}^{-1/2}
        \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(k-1)} + \alpha
        \mathbf{X}^{(0)}

        \mathbf{X}^{\prime} &= \mathbf{X}^{(K)},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        K (int): Number of iterations :math:`K`.
        alpha (float): Teleport probability :math:`\alpha`.
        dropout (float, optional): Dropout probability of edges during
            training. (default: :obj:`0`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, alpha: float, lower: float, upper: float,  homophily: bool, nameFunc: str, namePoly: str, Vandermonde: str, dropout: float = 0.,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.homophily = homophily
        self.Vandermonde = Vandermonde
        self.lower = lower
        self.upper = upper
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        if(nameFunc == 'g_0'):
            self.coeffs = compare_fit_panelA(g_0, namePoly, Vandermonde, self.K, self.lower, self.upper)
        elif(nameFunc == 'g_1'):
            self.coeffs = compare_fit_panelA(g_1, namePoly, Vandermonde, self.K, self.lower, self.upper) 
        elif(nameFunc == 'g_2'):
            self.coeffs = compare_fit_panelA(g_2,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_3'):
            self.coeffs = compare_fit_panelA(g_3,namePoly, Vandermonde, self.K, self.lower, self.upper)
        elif(nameFunc == 'g_4'):
            self.coeffs = compare_fit_panelA(g_4,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_band_rejection'):
            self.coeffs = compare_fit_panelA(g_band_rejection,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_band_pass'):
            self.coeffs = compare_fit_panelA(g_band_pass,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_low_pass'):
            self.coeffs = compare_fit_panelA(g_low_pass,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_high_pass'):
            self.coeffs = compare_fit_panelA(g_high_pass,namePoly, Vandermonde, self.K,self.lower, self.upper)
        elif(nameFunc == 'g_comb'):
            self.coeffs = compare_fit_panelA(g_comb,namePoly, Vandermonde, self.K,self.lower, self.upper)
        else:
            self.coeffs = compare_fit_panelA(g_fullRWR,namePoly, Vandermonde,self.K,self.lower, self.upper)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        h = x
        # Here this code could be more efficient 
        myb = self.coeffs[self.K-1]*x
        for k in range(self.K-2,-1,-1):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            #x = x * (1 - self.alpha)
            if (self.homophily):
                x = x + self.coeffs[k]*myb
            else:
                x = self.coeffs[k]*x + myb

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'




def generateCoeff( K, Init, nameFunc, homophily, Vandermonde, lower, upper, Threeterm):

    assert Init in ['Monomial', 'Chebyshev', 'Legendre', 'Jacobi', 'PPR','SChebyshev']
    if Init == 'Monomial':
        # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
        #x = m_polynomial_zeros(-(self.alpha), (self.alpha), self.K)
        if(nameFunc == 'g_0'):
            coeffs =  compare_fit_panelA(g_0, Init, Vandermonde, Threeterm,K, lower, upper) # m_polynomial_zeros(-(self.alpha), (self.alpha), self.K)
        elif(nameFunc == 'g_1'):
            coeffs =  compare_fit_panelA(g_1, Init, Vandermonde, Threeterm, K, lower, upper) 
        elif(nameFunc == 'g_2'):
            coeffs =  compare_fit_panelA(g_2,Init,Vandermonde, Threeterm, K,lower, upper)
        elif(nameFunc == 'g_3'):
            coeffs =  compare_fit_panelA(g_3,Init,Vandermonde, Threeterm, K,lower, upper)
        elif(nameFunc == 'g_4'):
            coeffs = compare_fit_panelA(g_4,Init,Vandermonde,Threeterm, K,lower, upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
        elif(nameFunc == 'g_band_rejection'):
            coeffs = compare_fit_panelA(g_band_rejection,Init,Vandermonde,Threeterm, K,lower, upper)
        elif(nameFunc == 'g_band_pass'):
            coeffs = compare_fit_panelA(g_band_pass,Init,Vandermonde,Threeterm, K,lower, upper)
        elif(nameFunc == 'g_low_pass'):
            coeffs = compare_fit_panelA(g_low_pass,Init,Vandermonde,Threeterm, K,lower, upper)
        elif(nameFunc == 'g_high_pass'):
            coeffs = compare_fit_panelA(g_high_pass,Init,Vandermonde,Threeterm, K,lower, upper)
        elif(nameFunc == 'g_comb'):
           coeffs = compare_fit_panelA(g_comb,Init,Vandermonde,Threeterm, K,lower, upper)
        else:
            coeffs = compare_fit_panelA(g_fullRWR,Init,Vandermonde,Threeterm,K,lower, upper)
        #l = [i for i in range (1, len(coeffs)+1) ]
        coeffs = filter_jackson(coeffs)
        TEMP = coeffs
        
        # TEMP = p_polynomial_zeros(self.K)
        # TEMP = j_polynomial_zeros(self.K,0,1)
    elif Init == 'Chebyshev':
        # PPR-like
        if(nameFunc == 'g_0'):
            coeffs = compare_fit_panelA(g_0, Init, Vandermonde,Threeterm, K,lower, upper)
        elif(nameFunc == 'g_1'):
            coeffs = compare_fit_panelA(g_1, Init, Vandermonde,Threeterm, K,lower, upper) 
        elif(nameFunc == 'g_2'):
            coeffs = compare_fit_panelA(g_2,Init,Vandermonde,Threeterm, K,lower, upper)
        elif(nameFunc == 'g_3'):
            coeffs = compare_fit_panelA(g_3,Init,Vandermonde,Threeterm, K,lower, upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
        elif(nameFunc == 'g_4'):
            coeffs = compare_fit_panelA(g_4,Init,Vandermonde,Threeterm, K,lower, upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
        elif(nameFunc == 'g_band_rejection'):
            coeffs = compare_fit_panelA(g_band_rejection,Init,Vandermonde,Threeterm, K,lower, upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
        elif(nameFunc == 'g_band_pass'):
            coeffs = compare_fit_panelA(g_band_pass,Init,Vandermonde,Threeterm, K,lower, upper)
        elif(nameFunc == 'g_low_pass'):
            coeffs = compare_fit_panelA(g_low_pass,Init,Vandermonde,Threeterm, K,lower, upper)
        elif(nameFunc == 'g_high_pass'):
            coeffs = compare_fit_panelA(g_high_pass,Init,Vandermonde,Threeterm,K,lower, upper)
        elif(nameFunc == 'g_comb'):
            coeffs = compare_fit_panelA(g_comb,Init,Vandermonde,Threeterm,K,lower, upper)
        else:
            coeffs = compare_fit_panelA(g_fullRWR,Init, Vandermonde,K)
        coeffs = filter_jackson(coeffs)
        
        
        TEMP = coeffs
        #TEMP = t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)
    elif Init == 'Legendre':
        #TEMP = p_polynomial_zeros(self.K)
        if(nameFunc == 'g_0'):
            coeffs = compare_fit_panelA(g_0, Init, Vandermonde, Threeterm,K,lower, upper)#p_polynomial_zeros(self.K) 
        elif(nameFunc == 'g_1'):
            coeffs = compare_fit_panelA(g_1, Init, Vandermonde,Threeterm,K,lower, upper) #p_polynomial_zeros(self.K)
        elif(nameFunc == 'g_2'):
            coeffs = compare_fit_panelA(g_2,Init,Vandermonde, Threeterm,K,lower, upper)
        elif(nameFunc == 'g_3'):
            coeffs = compare_fit_panelA(g_3,Init,Vandermonde, Threeterm,K,lower, upper)#p_polynomial_zeros(K)
        elif(nameFunc == 'g_4'):
            coeffs = compare_fit_panelA(g_4,Init,Vandermonde,Threeterm,K,lower, upper)#t_polynomial_zeros(-(alpha), (alpha), K)#
        elif(nameFunc == 'g_band_rejection'):
            coeffs = compare_fit_panelA(g_band_rejection,Init,Vandermonde,Threeterm,K,lower, upper)#t_polynomial_zeros(-(alpha), (alpha), K)#
        elif(nameFunc == 'g_band_pass'):
            coeffs = compare_fit_panelA(g_band_pass,Init,Vandermonde,Threeterm,K,lower, upper)
        elif(nameFunc == 'g_low_pass'):
            coeffs = compare_fit_panelA(g_low_pass,Init,Vandermonde,Threeterm,K,lower, upper)
        elif(nameFunc == 'g_high_pass'):
            coeffs = compare_fit_panelA(g_high_pass,Init,Vandermonde,Threeterm,K,lower, upper)
        elif(nameFunc == 'g_comb'):
            coeffs = compare_fit_panelA(g_comb,Init,Vandermonde,Threeterm,K,lower, upper)
        else:
            coeffs = compare_fit_panelA(g_fullRWR,Init,K,lower, upper)
        
        coeffs = filter_jackson(coeffs)
        #coeffs = np.divide(coeffs, l)
        #coeffs = np.divide(coeffs, division)
        
        TEMP = coeffs
    elif Init == 'Jacobi':
        if(nameFunc == 'g_0'):
            coeffs = compare_fit_panelA(g_0, Init, Vandermonde, Threeterm,K,lower, upper)#p_polynomial_zeros(K) 
        elif(nameFunc == 'g_1'):
            coeffs = compare_fit_panelA(g_1, Init, Vandermonde,Threeterm,K,lower, upper) #p_polynomial_zeros(K)
        elif(nameFunc == 'g_2'):
            coeffs = compare_fit_panelA(g_2,Init,Vandermonde, Threeterm,K,lower, upper)
        elif(nameFunc == 'g_3'):
            coeffs = compare_fit_panelA(g_3,Init,Vandermonde, Threeterm,K,lower, upper)#p_polynomial_zeros(K)
        elif(nameFunc == 'g_4'):
            coeffs = compare_fit_panelA(g_4,Init,Vandermonde,Threeterm,K,lower, upper)#t_polynomial_zeros(-(alpha), (alpha), K)#
        elif(nameFunc == 'g_band_rejection'):
            coeffs = compare_fit_panelA(g_band_rejection,Init,Vandermonde,Threeterm,K,lower, upper)#t_polynomial_zeros(-(alpha), (alpha), K)#
        elif(nameFunc == 'g_band_pass'):
            coeffs = compare_fit_panelA(g_band_pass,Init,Vandermonde,Threeterm,K,lower, upper)
        elif(nameFunc == 'g_low_pass'):
            coeffs = compare_fit_panelA(g_low_pass,Init,Vandermonde,Threeterm,K,lower, upper)
        elif(nameFunc == 'g_high_pass'):
            coeffs = compare_fit_panelA(g_high_pass,Init,Vandermonde,Threeterm,K,lower, upper)
        elif(nameFunc == 'g_comb'):
            coeffs = compare_fit_panelA(g_comb,Init,Vandermonde,Threeterm,K,lower, upper)
        else:
            coeffs = compare_fit_panelA(g_fullRWR,Init,K)

        
        #coeffs = np.divide(coeffs, division)
        TEMP = coeffs
        #TEMP = j_polynomial_zeros(self.K,0,1)

    temp = Parameter(torch.tensor(TEMP))
    
    
    coe_tmp = torch.flip(temp, dims=(0,))
    coe_tmp=temp
    coe=coe_tmp.clone()
    return coe

# Global Attention Mechanism
class feature_attention(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=5, rate=4):
        super(feature_attention, self).__init__()
        self.nconv = nn.Conv2d(input_dim, output_dim, kernel_size=(1, 1))
        self.channel_attention = nn.Sequential(  
            nn.Linear(output_dim, int(output_dim / rate)),  
            nn.ReLU(inplace=True),  
            nn.Linear(int(output_dim / rate), output_dim)  
        )
        self.spatial_attention = nn.Sequential(  
            nn.Conv2d(output_dim, int(output_dim / rate), kernel_size=(1, kernel_size), padding=(0, (kernel_size - 1) // 2)),  
            nn.BatchNorm2d(int(output_dim / rate)),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(int(output_dim / rate), output_dim, kernel_size=(1, kernel_size), padding=(0, (kernel_size - 1) // 2)),  
            nn.BatchNorm2d(output_dim)  
        )
        
    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  # [B, D, N, T]
        x = self.nconv(x)  # 扩展数据的特征维度
        b, c, n, t = x.shape  
        x_permute = x.permute(0, 2, 3, 1)  # [B, N, T, C]
        x_att_permute = self.channel_attention(x_permute) 
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)  # [B, C, N, T]
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out.permute(0, 3, 2, 1)

class AVWGCN(nn.Module):
    def __init__(self, in_dim, out_dim, cheb_k, embed_dim):
        """
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param cheb_k: 切比雪夫多项式的阶，默认为3
        :param embed_dim: 节点的嵌入维度
        """
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, in_dim, out_dim))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, out_dim))
        alpha = 0.0
        beta = 0.0

    def forward(self, x, node_embedding):
        """
        :param x: (B, N, C_in)
        :param node_embedding: (N, D), 这里的node_embedding是可学习的
        :return: (B, N, C_out)
        """
        node_num = node_embedding.shape[0]
        # 自适应的学习节点间的内s在隐藏关联获取邻接矩阵
        # D^(-1/2)AD^(-1/2)=softmax(ReLU(E * E^T)) - (N, N)
        coeffs = generateCoeff(11, 'Chebyshev', 'g_0', False, False, -0.9, 0.9, True)
        support = F.softmax(F.relu(torch.mm(node_embedding, node_embedding.transpose(0, 1))), dim=1)
        if ALGO == 'Garnoldi':
          support = coeffs[0] * support
        
        #support = coeffs[0]*support # Open when running Garnoldi
        # 这里得到的support表示标准化的拉普拉斯矩阵
        support_set = [torch.eye(node_num).to(support.device), support]
        for k in range(2, self.cheb_k):
            # Z(k) = 2 * L * Z(k-1) - Z(k-2)
            if ALGO == 'Garnoldi':
              #Chebyshev
              support_set.append(torch.matmul(2 * coeffs[k] * support, support_set[-1]) - support_set[-2])
              
              #Monomial
              #support_set.append(torch.matmul(support*coeffs[k], support_set[-1]))
              
              #Legendre
              #a = (2 * k - 1) / k
              #b = (k - 1) / k
              #support_set.append(torch.matmul(a * support*coeffs[k], support_set[-1]) - b * support_set[-2])

              #Jacobi
              #a = (2 * k + alpha + beta - 1) * (2 * k + alpha + beta) / (2 * k * (k + alpha + beta))
              #b = (k + alpha - 1) * (k + beta - 1) * (2 * k + alpha + beta) / (2 * k * (k + alpha + beta) * (2 * k + alpha + beta - 2))
              #support_set.append(torch.matmul(a * support*coeffs[k], support_set[-1]) - b * support_set[-2])
            else:
              #Chebyshev
              support_set.append(torch.matmul(2 * support, support_set[-1]) - support_set[-2])
              
              #Monomial
              #support_set.append(torch.matmul(support, support_set[-1]))
              
              #Legendre
              #a = (2 * k - 1) / k
              #b = (k - 1) / k
              #support_set.append(torch.matmul(a * support, support_set[-1]) - b * support_set[-2])

              #Jacobi
              #a = (2 * k + alpha + beta - 1) * (2 * k + alpha + beta) / (2 * k * (k + alpha + beta))
              #b = (k + alpha - 1) * (k + beta - 1) * (2 * k + alpha + beta) / (2 * k * (k + alpha + beta) * (2 * k + alpha + beta - 2))
              #support_set.append(torch.matmul(a * support, support_set[-1]) - b * support_set[-2])


        supports = torch.stack(support_set, dim=0) # (K, N, N)
        # (N, D) * (D, K, C_in, C_out) -> (N, K, C_in, C_out)
        weights = torch.einsum('nd, dkio->nkio', node_embedding, self.weights_pool)
        # (N, D) * (D, C_out) -> (N, C_out)
        bias = torch.matmul(node_embedding, self.bias_pool)

        # 多阶切比雪夫计算：(K, N, N) * (B, N, C_in) -> (B, K, N, C_in)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # (B, K, N, C_in) 很好奇为什么不在dim=1相加?
        x_g = x_g.permute(0, 2, 1, 3)  # (B, N, K, C_in) * (N, K, C_in, C_out)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # (B, N, C_out)
        return x_gconv

class AGCRNCell(nn.Module):
    def __init__(self, num_node, in_dim, out_dim, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.num_node = num_node
        self.hidden_dim = out_dim
        self.gate = AVWGCN(in_dim + out_dim, 2 * out_dim, cheb_k, embed_dim)
        self.update = AVWGCN(in_dim + out_dim, out_dim, cheb_k, embed_dim)

    def forward(self, x, state, node_embedding):
        # x: (B, N, C), state: (B, N, D)
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        # 两个门控 forget、update
        z_r = torch.sigmoid(self.gate(input_and_state, node_embedding))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, r*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embedding))
        h = z * state + (1 - z) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_node, self.hidden_dim)

class AVWDCRNN(nn.Module):
    def __init__(self, num_node, in_dim, out_dim, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, "At least one DCRNN layer in the Encoder."
        self.num_node = num_node
        self.input_dim = in_dim
        self.num_layers = num_layers
        self.dcrnnn_cells = nn.ModuleList()
        self.dcrnnn_cells.append(AGCRNCell(num_node, in_dim, out_dim, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnnn_cells.append(AGCRNCell(num_node, out_dim, out_dim, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embedding):
        """
        :param x: (B, T, N, in_dim)
        :param init_state: (num_layers, B, N, hidden_dim)
        :param node_embedding: (N, D)
        :return:
        """
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnnn_cells[i](current_inputs[:, t, :, :], state, node_embedding)
                inner_states.append(state)
            output_hidden.append(state)  # 最后一个时间步输出的隐藏状态
            current_inputs = torch.stack(inner_states, dim=1) # (B, T, N, hid_dim)

        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        output_hidden = torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []  # 初始化隐藏层
        for i in range(self.num_layers):
            init_states.append(self.dcrnnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, out_dim, max_len=1):
        super(PositionalEncoding, self).__init__()

        # compute the positional encodings once in log space.
        pe = torch.zeros(max_len, out_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_dim, 2) *
                             - math.log(10000.0) / out_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (B, T, N, D) + (1, T, 1, D)
        x = x + Variable(self.pe.to(x.device), requires_grad=False)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        # 计算在时间维度上的多头注意力机制
        self.positional_encoding = PositionalEncoding(embed_size)
        self.embed_size = embed_size
        self.heads = heads
        # 要求嵌入层特征维度可以被heads整除
        assert embed_size % heads == 0
        self.head_dim = embed_size // heads   # every head dimension

        self.W_V = nn.Linear(self.embed_size, self.head_dim * heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * heads, bias=False)
        # LayerNorm在特征维度上操作
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size)
        )

    def forward(self, x):
        """
        :param x: [B, T, N, D]
        """
        batch_size, _, _, d_k = x.shape
        x = self.positional_encoding(x).permute(0, 2, 1, 3)   # [B, N, T, D]
        # 计算Attention的Q、K、V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        Q = torch.cat(torch.split(Q, self.head_dim, dim=-1), dim=0)  # [k*B, N, T, d_k]
        K = torch.cat(torch.split(K, self.head_dim, dim=-1), dim=0)  # [k*B, N, T, d_k]
        V = torch.cat(torch.split(V, self.head_dim, dim=-1), dim=0)
        # 考虑上下文的长期依赖信息
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float))
        attention = F.softmax(scores, dim=-1)  # [k * B, N, T, T]
        context = torch.matmul(attention, V)   # context vector
        context = torch.cat(torch.split(context, batch_size, dim=0), dim=-1)
        context = context + x    # residual connection
        out = self.norm1(context)
        out = self.fc(out) + context  # residual connection
        out = self.norm2(out)
        return out

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, adj, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.adj = adj
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        # h: (B, T, N, D)
        Wh = torch.matmul(h, self.W)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.permute(0, 1, 3, 2)
        e = self.leakyrelu(e)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(self.adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        out = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(out)
        else:
            return out


class GPR_prop(MessagePassing):
    def __init__(self, K, alpha, Init, Gamma=None, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.Init = Init
        self.Gamma = Gamma

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            TEMP = torch.zeros(K + 1)
            TEMP[int(alpha)] = 1.0
        elif Init == 'PPR':
            TEMP = alpha * (1 - alpha) ** torch.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            TEMP = (alpha) ** torch.arange(K + 1)
            TEMP = TEMP / torch.sum(torch.abs(TEMP))
        elif Init == 'Random':
            bound = torch.sqrt(torch.tensor(3.0 / (K + 1)))
            TEMP = torch.rand(K + 1) * 2 * bound - bound
            TEMP = TEMP / torch.sum(torch.abs(TEMP))
        elif Init == 'WS':
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index):
        edge_index, norm = gcn_norm(edge_index, num_nodes=x.size(1), dtype=x.dtype)
        # edge_index, norm = custom_gcn_norm(edge_index, num_nodes=x.size(1), dtype=x.dtype)
        hidden = x * self.temp[0]
        x = x.T
        hidden = hidden.T
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            # x = self.custom_propagate(edge_index, x=x, norm=norm)
            # x = self.custom_propagate(edge_index, x=x.T, norm=norm)
            gamma = self.temp[k + 1]
            # hidden = hidden + gamma * x
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def _repr_(self):
        return '{}(K={}, temp={})'.format(self._class.name_, self.K, self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, num_node, input_dim, output_dim, hidden, cheb_k, num_layers, embed_dim):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(768, 64)  # (hidden_dim*num_nodes, hidden_dim) 19, 1
        self.lin2 = Linear(64, 768)

        self.prop1 = GPR_prop(cheb_k, 0.5, 'PPR', None)

        self.dprate = 0.5
        self.dropout = 0.2
        self.num_layers = num_layers
        ###
        self.dcrnnn_cells = nn.ModuleList()
        self.dcrnnn_cells.append(AGCRNCell(num_node, input_dim, output_dim, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnnn_cells.append(AGCRNCell(num_node, input_dim, output_dim, cheb_k, embed_dim))

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x):
        edge_index = read_edge_list_csv()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.to('cpu')
        x_reshaped = x.reshape(x.size(0), -1)  # -1 infers the remaining dimension based on the input shape

        # Apply linear layer
        x = F.relu(self.lin1(x_reshaped))
        # x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            x = x.transpose(0, 1)

            # x: (B, T, N, hidden_dim)
            # Reshape it from (5, 1216) to (5, 1, 19, 64)
            x = x.view(x.size(0), 1, 12, 64)  # Manually reshape to (5, 1, 19, 64)
          

            # Apply log softmax along the appropriate dimension
            x = F.log_softmax(x, dim=3)  # Assuming the last dimension (64) is the one to apply softmax to
            return x

    def init_hidden(self, batch_size):
        """
        Initialize hidden states for all layers.

        Args:
        - batch_size (int): The batch size for the input data.

        Returns:
        - init_states (Tensor): Initialized hidden states for all layers.
        """
        init_states = []  # Initialize hidden states list for all layers
        for i in range(self.num_layers):
            # Assuming each cell in dcrnnn_cells has an init_hidden_state method
            init_states.append(self.dcrnnn_cells[i].init_hidden_state(batch_size))

        # Stack the initialized states along the first dimension to get (num_layers, B, N, hidden_dim)
        return torch.stack(init_states, dim=0)


##############################################################
class APPNP(MessagePassing):
    def __init__(self, K, alpha, dropout=0.,
                 cached=False, add_self_loops=True,
                 normalize=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        super().reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_weight=None):
        # print("APPNP.forward - Initial x:", x.size())
        # print("APPNP.forward - Initial edge_index:", edge_index.size())
        nodes = 19
        if self.normalize:
            if isinstance(edge_index, torch.Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index = read_edge_list_csv()
                    # print("Edge index:", edge_index)
                    # print("Number of nodes:", (int)(x.size(1) / 64))
                    edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes=(x.size(1)),
                                                       dtype=x.dtype)
                    # edge_index = torch.tensor([
                    # [3, 2, 0, 1, 1, 7, 6, 4, 5, 5, 8, 11, 12, 11, 10, 9, 9, 13, 10, 14, 17, 17, 18, 16],
                    # [2, 1, 1, 6, 7, 4, 5, 8, 8, 11, 12, 12, 9, 10, 9, 13, 14, 14, 17, 18, 18, 16, 15, 15]
                    # ])
                    # print("APP Edge index shape:", edge_index.shape)
                    # print("Edge index content:", edge_index)
                    # print("Edge weight shape:", edge_weight.shape)
                    # print("APP Edge weight content:", edge_weight)
                    if self.cached:
                        self._cached_edge_index = edge_index
                else:
                    edge_index = cache
                    # edge_index = torch.tensor([
            # [3, 2, 0, 1, 1, 7, 6, 4, 5, 5, 8, 11, 12, 11, 10, 9, 9, 13, 10, 14, 17, 17, 18, 16],
            # [2, 1, 1, 6, 7, 4, 5, 8, 8, 11, 12, 12, 9, 10, 9, 13, 14, 14, 17, 18, 18, 16, 15, 15]
            # ])

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(edge_index, num_nodes=x.size(1), dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # print("APPNP.forward - Normalized edge_index:", edge_index.size())
        x = x.T
        h = x

        for k in range(self.K):
            # print(f"APPNP.forward - Iteration {k}, x size:", x.size())
            if self.dropout > 0 and self.training:
                x = F.dropout(x, p=self.dropout, training=self.training)
                # print(f"APPNP.forward - After dropout, x size:", x.size())

            # propagate_type: (x: Tensor)
            x = self.propagate(edge_index, x=x)
            # print(f"APPNP.forward - After propagate, x size:", x.size())

            x = x * (1 - self.alpha)
            # print("Shape of x:", x.shape)
            # print("Shape of h:", h.shape)
            # h = h.T
            x = x + self.alpha * h

        # print("APPNP.forward - Final x size:", x.size())
        #x = x.T
        # h = x
        return x

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'


class APPNP_Net(torch.nn.Module):
    def __init__(self, num_node, input_dim, output_dim, hidden, cheb_k, num_layers, embed_dim):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(768, 64)  # (512, 64) for Konya & (1216,64) for Kcetas
        self.lin2 = Linear(64, 768)
        self.prop1 = APPNP(cheb_k, 0.5, 0.2, False, True, True)
        self.dropout = 0.2
        self.num_layers = num_layers
        self.dcrnnn_cells = nn.ModuleList()
        self.dcrnnn_cells.append(AGCRNCell(num_node, input_dim, output_dim, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnnn_cells.append(AGCRNCell(num_node, input_dim, output_dim, cheb_k, embed_dim))

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        edge_index = read_edge_list_csv()
        # edge_index, norm = gcn_norm(edge_index, num_nodes=x.size(1), dtype=x.dtype)

        # print(edge_index)
        # print("Initial x:", x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.to('cpu')

        # Reshape the input
        x_reshaped = x.reshape(x.size(0), -1)  # -1 infers the remaining dimension based on the input shape
        # print("x reshaped to:", x_reshaped.size())

        x = F.relu(self.lin1(x_reshaped))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # print("After linear and dropout, x size:", x.size())
        # print("Edge index:", edge_index)

        x = self.lin2(x)
        # print("After second linear layer, x size:", x.size())

        x = self.prop1(x, edge_index)
        # print("After propagation, x size:", x.size())
        x = x.transpose(0, 1)
        # Reshape it from (5, 1216) to (5, 1, 19, 64) for Kcetas
        # (5, 1, 8, 64) for Konya
        x = x.reshape(x.size(0), 1, 12, 64)  # Manually reshape to (5, 1, 19, 64)
        # print("After reshaping, x size:", x.size())

        # Apply log softmax along the appropriate dimension
        x = F.log_softmax(x, dim=3)  # Assuming the last dimension (64) is the one to apply softmax to
        # print("After log_softmax, x size:", x.size())
        return x

    def init_hidden(self, batch_size):
        """
        Initialize hidden states for all layers.

        Args:
        - batch_size (int): The batch size for the input data.

        Returns:
        - init_states (Tensor): Initialized hidden states for all layers.
        """
        init_states = []  # Initialize hidden states list for all layers
        for i in range(self.num_layers):
            # Assuming each cell in dcrnnn_cells has an init_hidden_state method
            init_states.append(self.dcrnnn_cells[i].init_hidden_state(batch_size))

        # Stack the initialized states along the first dimension to get (num_layers, B, N, hidden_dim)
        return torch.stack(init_states, dim=0)

####################################################################
def read_edge_list_csv():
    # Read the CSV file into a DataFrame
    df = pd.read_csv(GRAPH)

    # Extract the 'from' and 'to' columns as numpy arrays
    edges_from = df['from'].to_numpy()
    edges_to = df['to'].to_numpy()

    # Create the edge index tensor
    edge_index = torch.tensor([edges_from, edges_to], dtype=torch.long)

    return edge_index



class Model(nn.Module):
    def __init__(self, num_node, input_dim, hidden_dim, output_dim, embed_dim, cheb_k, horizon, num_layers, heads, timesteps, A, kernel_size):
        super(Model, self).__init__()
        self.A = A
        self.timesteps = timesteps
        self.num_node = num_node
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.num_layers = num_layers

        # node embed
        self.node_embedding = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)
        # encoder
        self.feature_attention = feature_attention(input_dim=input_dim, output_dim=hidden_dim, kernel_size=kernel_size)
        if ALGO in ['Garnoldi', 'default']:
            self.encoder = AVWDCRNN(num_node, hidden_dim, hidden_dim, cheb_k, embed_dim, num_layers)
        else:
            if ALGO == 'APPNP':
                self.encoder = APPNP_Net(num_node, input_dim, output_dim, hidden_dim, cheb_k, num_layers, embed_dim)
            elif ALGO == 'GPRGNN':
                self.encoder = GPRGNN(num_node, input_dim, output_dim, hidden_dim, cheb_k, num_layers, embed_dim)
        self.GraphAttentionLayer = GraphAttentionLayer(hidden_dim, hidden_dim, A, dropout=0.5, alpha=0.2, concat=True)
        self.MultiHeadAttention = MultiHeadAttention(embed_size=hidden_dim, heads=heads)
        # predict
        self.nconv = nn.Conv2d(1, self.horizon, kernel_size=(1, 1), bias=True)
        self.end_conv = nn.Conv2d(hidden_dim, 1, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        # x: (B, T, N, D)
        batch_size = x.shape[0]
        x = self.feature_attention(x)
        init_state = self.encoder.init_hidden(batch_size)
        if ALGO in ['Garnoldi', 'default']:
            init_state = self.encoder.init_hidden(batch_size)
            output, _ = self.encoder(x, init_state, self.node_embedding)
        else:
            output = self.encoder(x)

        #output, _ = self.encoder(x, init_state, self.node_embedding)  # (B, T, N, hidden_dim)
        #output = self.encoder(x)
        state = output[:, -1:, :, :]
        state = self.nconv(state)
        SAtt = self.GraphAttentionLayer(state)
        TAtt = self.MultiHeadAttention(output).permute(0, 2, 1, 3)
        out = SAtt + TAtt
        out = self.end_conv(out.permute(0, 3, 2, 1))  # [B, 1, N, T] -> [B, N, T]
        out = out.permute(0, 3, 2, 1)   # [B, T, N]
        return out
