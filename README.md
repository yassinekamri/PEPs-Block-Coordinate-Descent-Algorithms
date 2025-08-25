# PEPs for the Worst-Case Analysis of Cyclic Block Coordinate Descent-Type Algorithms

This repository contains the code to reproduce the results of the following paper:  

**Yassine Kamri, François Glineur, Julien M. Hendrickx, and Ion Necoara**  
[*On the Worst-Case Analysis of Cyclic Block Coordinate Descent-Type Algorithms*](https://arxiv.org/abs/2507.16675).  
arXiv preprint arXiv:2507.16675, 2025.

---

## Authors
- Yassine Kamri  
- Julien M. Hendrickx  
- François Glineur  
- Ion Necoara  

---

## Getting Started
The code is written in **Julia** and requires:  
- The [JuMP](https://jump.dev) optimization toolbox  
- The [Mosek](https://www.mosek.com) SDP solver  

---

## File Description

The file **`PEP_block_coordinates_algorithms.jl`** contains the following functions:

- **`pep_ccd_settingALL`**  
  Computes an upper bound on the worst-case convergence of *Cyclic Coordinate Descent (CCD)* using the PEP framework in **Setting ALL** (see article for the definition).

- **`pep_ccd_settingINIT`**  
  Computes an upper bound on the worst-case convergence of CCD using the PEP framework in **Setting INIT** (see article for the definition).

- **`pep_alternating_minimization`**  
  Computes an upper bound on the worst-case convergence of *Alternating Minimization (AM)* using the PEP framework.

- **`pep_cacd`**  
  Computes an upper bound on the worst-case convergence of *Cyclic Accelerated Coordinate Descent (CACD)* using the PEP framework.

- **`pep_ccd_HDZ`**  
  Computes an upper bound on the worst-case convergence of CCD in **Setting ALL** using the PEP framework described in:  
  *Abbaszadehpeivasti, H., de Klerk, E., & Zamani, M. (2023). Convergence rate analysis of randomized and cyclic coordinate descent for convex optimization through semidefinite programming. Applied Set-Valued Analysis and Optimization, 5(2), 141–153.*  
  [DOI link](https://doi.org/10.23952/asvao.5.2023.2.02)

- **`pep_ccd_lb`**  
  Computes a lower bound on the worst-case convergence of CCD using the PEP framework described in:  
  *Y. Kamri, J. M. Hendrickx, and F. Glineur, "On the Worst-Case Analysis of Cyclic Coordinate-Wise Algorithms on Smooth Convex Functions," 2023 European Control Conference (ECC), Bucharest, Romania, 2023, pp. 1–8.*  
  [IEEE link](https://ieeexplore.ieee.org/document/10178198)  
