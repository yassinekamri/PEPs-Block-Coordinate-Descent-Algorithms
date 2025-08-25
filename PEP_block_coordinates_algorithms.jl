# ============================================================================
# PEP for block coordinate descent (BCD) algorithms
# ----------------------------------------------------------------------------
# Julia code to compute bounds on the convergence of several BCD
# algorithms using the PEP framework  described in:
#   Y. Kamri, F. Glineur, J. M. Hendrickx, and I. Necoara.
#   "On the Worst-Case Analysis of Cyclic Block Coordinate Descent type Algorithms." arXiv, 2025.
#   Link: https://arxiv.org/abs/2507.16675
#
#
# Dependencies:
#   - JuMP, MosekTools, Mosek
#   - LinearAlgebra
#   - ProgressBars (optional)
#   - JLD2 (optional)
# ============================================================================
#Imports
using JuMP
using MosekTools
using LinearAlgebra
using ProgressBars
using JLD2
# ---------------------------------------------------------------------------
# PEP formulation for CCD in Setting ALL
# ---------------------------------------------------------------------------
# Computes the convex formulation of the PEP for cyclic coordinate descent
# for setting ALL.
# See Kamri et al., "On the Worst-Case Analysis of Cyclic Block Coordinate
# Descent Type Algorithms" (2025) for theoretical background.
function pep_ccd_settingALL(K, nblocks, L, h)

    # K: total number of iterates for CCD
    # nblocks: number of blocks for CCD
    # L: vector od smoothness constants
    # h: step size

    dimG = K + 2
    dimF = K + 1

    # xbar, gbar represents resêctively iterates and associated gradients
    xbar = [zeros(dimG) for _ in 1:(K+2), __ in 1:nblocks]
    gbar = [zeros(dimG) for _ in 1:(K+2), __ in 1:nblocks]

    for i in 1:nblocks
        xbar[1, i][1] = 1.0
    end
    for j in 1:(K+2), i in 1:nblocks
        if j <= K + 1
            gbar[j, i][j + 1] = 1.0
        end
    end

    fbar = hcat(Matrix(I, dimF, dimF), zeros(dimF, 1)) # represents functional values

    # CCD updates
    for i in 1:K
        idx = mod(i, nblocks) + 1
        for j in 1:nblocks
            if j == idx
                xbar[i + 1, j] = xbar[i, j] .- (h / L[j]) .* gbar[i, j]
            else
                xbar[i + 1, j] = xbar[i, j]
            end
        end
    end

    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Variables:
    # F: functional values f_i = F' * fbar[:,i] 
    # G: Gram matrices for each block of coordinates
    # |x^(i)_k|^2 = xbar[k, i]'  * G[i] * xbar[k, i]
    # |g^(i)_k|^2 = gbar[k, i]' * G[i] * gbar[k, i]
    # g^(i)_k * x^(i)_j = gbar[k, i]' * G[i] * xbar[j, i]
    G = [@variable(model, [1:dimG, 1:dimG], PSD) for _ in 1:nblocks]
    @variable(model, F[1:dimF])

    # Inital condition of Setting ALL. (see paper https://arxiv.org/abs/2507.16675) 
    for i = 1:1+K
        if mod(i,nblocks) == 1
            condinit = 0
            for j = 1:nblocks
                condinit = condinit + xbar[i,j]'*G[j]*xbar[i,j]
            end
            @constraint(model, condinit <= 1.0)
        end
    end

    # Interpolation constraints for L-coordinate-wise smooth convex functions (see paper https://arxiv.org/abs/2507.16675) 
    for i in 1:(K+2)
        for j in 1:(K+2)
            if i != j
                fi = fbar[:, i]
                fj = fbar[:, j]

                cond = F' * (fj - fi)
                for k in 1:nblocks
                    cond += gbar[j, k]' * G[k] * (xbar[i, k] - xbar[j, k])
                end

                for t in 1:nblocks
                    dgt = gbar[i, t] - gbar[j, t]
                    condt = cond + (1 / (2 * L[t])) * (dgt' * G[t] * dgt)
                    @constraint(model, condt <= 0.0)
                end
            end
        end
    end

    # Objective: maximize f(x_K)-f(x_*)
    obj_vec = fbar[:, K + 1]
    @objective(model, Max, F' * obj_vec)

    optimize!(model)

    objective = objective_value(model)
    G_val = [value.(G[j]) for j in 1:nblocks]
    return objective, G_val
end

K = 2 # total number of iterates
nblocks = 2 # number of blocks
L = [1,1] # vector of smoothness constants
h = 1 # step-size
obj =  pep_ccd_settingALL(K, nblocks, L, h)[1]

# ---------------------------------------------------------------------------
# PEP formulation for CCD in Setting INIT
# ---------------------------------------------------------------------------
# Computes the convex formulation of the PEP for cyclic coordinate descent
# for setting INIT.
# See Kamri et al., "On the Worst-Case Analysis of Cyclic Block Coordinate
# Descent Type Algorithms" (2025) for theoretical background.
function pep_ccd_settingINIT(K, nblocks, L, h)

    # K: total number of iterates for CCD
    # nblocks: number of blocks for CCD
    # L: vector od smoothness constants
    # h: step size

    dimG = K + 2
    dimF = K + 1

    # xbar, gbar represents resêctively iterates and associated gradients
    xbar = [zeros(dimG) for _ in 1:(K+2), __ in 1:nblocks]
    gbar = [zeros(dimG) for _ in 1:(K+2), __ in 1:nblocks]

    for i in 1:nblocks
        xbar[1, i][1] = 1.0
    end
    for j in 1:(K+2), i in 1:nblocks
        if j <= K + 1
            gbar[j, i][j + 1] = 1.0
        end
    end

    fbar = hcat(Matrix(I, dimF, dimF), zeros(dimF, 1)) # represents functional values

    # CCD updates
    for i in 1:K
        idx = mod(i, nblocks) + 1
        for j in 1:nblocks
            if j == idx
                xbar[i + 1, j] = xbar[i, j] .- (h / L[j]) .* gbar[i, j]
            else
                xbar[i + 1, j] = xbar[i, j]
            end
        end
    end

    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Variables:
    # F: functional values f_i = F' * fbar[:,i] 
    # G: Gram matrices for each block of coordinates
    # |x^(i)_k|^2 = xbar[k, i]'  * G[i] * xbar[k, i]
    # |g^(i)_k|^2 = gbar[k, i]' * G[i] * gbar[k, i]
    # g^(i)_k * x^(i)_j = gbar[k, i]' * G[i] * xbar[j, i]
    G = [@variable(model, [1:dimG, 1:dimG], PSD) for _ in 1:nblocks]
    @variable(model, F[1:dimF])

    # Inital condition of Setting INIT. (see paper https://arxiv.org/abs/2507.16675) 
    init_expr = sum(L[j] * (xbar[1, j]' * G[j] * xbar[1, j]) for j in 1:nblocks)
    @constraint(model, init_expr <= 1.0)

    # Interpolation constraints for L-coordinate-wise smooth convex functions (see paper https://arxiv.org/abs/2507.16675) 
    for i in 1:(K+2)
        for j in 1:(K+2)
            if i != j
                fi = fbar[:, i]
                fj = fbar[:, j]

                cond = F' * (fj - fi)
                for k in 1:nblocks
                    cond += gbar[j, k]' * G[k] * (xbar[i, k] - xbar[j, k])
                end

                for t in 1:nblocks
                    dgt = gbar[i, t] - gbar[j, t]
                    condt = cond + (1 / (2 * L[t])) * (dgt' * G[t] * dgt)
                    @constraint(model, condt <= 0.0)
                end
            end
        end
    end

    # Objective: maximize f(x_K)-f(x_*)
    obj_vec = fbar[:, K + 1]
    @objective(model, Max, F' * obj_vec)

    optimize!(model)

    objective = objective_value(model)
    G_val = [value.(G[j]) for j in 1:nblocks]
    return objective, G_val
end

#--------------- test ----------------------------------

K = 2 # total number of iterates
nblocks = 2 # number of blocks
L = [1,1] # vector of smoothness constants
h = 1 # step-size
obj =  pep_ccd_settingINIT(K, nblocks, L, h)[1]


# ---------------------------------------------------------------------------
# PEP formulation for alternating minimization
# ---------------------------------------------------------------------------
# Computes the convex formulation of the PEP for 2-block alternating minimization
# See Kamri et al., "On the Worst-Case Analysis of Cyclic Block Coordinate
# Descent Type Algorithms" (2025) for theoretical background.
function pep_alternating_minimization(K, Lx, Ly)
    
    # K: Total number of alternating minimization steps
    # Lx: smoothness constant along the block of coordinates x
    # Ly: smoothness constant along the block of coordinates x

    dimG = 3K ÷ 2 + 2
    dimF = K + 1            

    # xbar, gbar represents resêctively iterates and associated gradients
    xbar  = zeros(dimG, K + 1)
    ybar  = zeros(dimG, K + 1)
    gxbar = zeros(dimG, K + 1)
    gybar = zeros(dimG, K + 1)

    
    gxbar[(K ÷ 2) + 2 : dimG, :] .= Matrix(I, K + 1, K + 1)
    gybar[(K ÷ 2) + 2 : dimG, :] .= Matrix(I, K + 1, K + 1)

    xbar[1, 1] = 1.0
    ybar[1, 1] = 1.0

    # alternating minimization updates
    idx = 1
    for i in 2:K+1
        if iseven(i)
            idx += 1
            xbar[idx, i] = 1.0
        else
            xbar[:, i] = xbar[:, i - 1]
        end
    end

    # alternating minimization updates
    idy = 1
    for i in 2:K+1
        if isodd(i)
            idy += 1
            ybar[idy, i] = 1.0
        else
            ybar[:, i] = ybar[:, i - 1]
        end
    end

   
    xbar  = hcat(xbar,  zeros(dimG))
    ybar  = hcat(ybar,  zeros(dimG))
    gxbar = hcat(gxbar, zeros(dimG))
    gybar = hcat(gybar, zeros(dimG))

   # alternating minimization updates
    for i in 2:K+1
        if iseven(i)
            gxbar[:, i] .= 0.0
        else
            gybar[:, i] .= 0.0
        end
    end

    # fbar : function values
    fbar = hcat(Matrix(I, K + 1, K + 1), zeros(K + 1))

    # Model
    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Variables:
    # F: functional values f_i = F' * fbar[:,i] 
    # Gx, Gy: Gram matrices for each block of coordinates
    # |x^(i)_k|^2 = xbar[k, i]'  * Gx * xbar[k, i]
    # |y^(i)_k|^2 = ybar[k, i]' * Gy * ybar[k, i]
    @variable(model, Gx[1:dimG, 1:dimG], PSD)
    @variable(model, Gy[1:dimG, 1:dimG], PSD)
    @variable(model, F[1:dimF])

    # Initialization constraints
    for i in 1:K+1
        if isodd(i)
            x = xbar[:, i]
            y = ybar[:, i]
            @constraint(model, x' * Gx * x + y' * Gy * y <= 1.0)
        end
    end

    # Interpolation constraints for coordinate-wise smooth convex functions
    for i in 1:K+2
        for j in 1:K+2
            if i != j
                xi  = xbar[:, i];   xj  = xbar[:, j]
                yi  = ybar[:, i];   yj  = ybar[:, j]
                gxi = gxbar[:, i];  gxj = gxbar[:, j]
                gyi = gybar[:, i];  gyj = gybar[:, j]
                fi  = fbar[:, i];   fj  = fbar[:, j]

                @constraint(model,
                    F' * (fj - fi) +
                    gxj' * Gx * (xi - xj) +
                    gyj' * Gy * (yi - yj) +
                    (1 / (2 * Lx)) * (gxi - gxj)' * Gx * (gxi - gxj) <= 0.0
                )

                @constraint(model,
                    F' * (fj - fi) +
                    gxj' * Gx * (xi - xj) +
                    gyj' * Gy * (yi - yj) +
                    (1 / (2 * Ly)) * (gyi - gyj)' * Gy * (gyi - gyj) <= 0.0
                )
            end
        end
    end

    # Objective: maximize f(x_K)-f(x_*)
    @objective(model, Max, F' * fbar[:, K + 1])

    optimize!(model)

    objective = objective_value(model)
    F_val  = value.(F)
    Gx_val = value.(Gx)
    Gy_val = value.(Gy)

    return objective, F_val, Gx_val, Gy_val, xbar, ybar, gxbar, gybar, fbar
end

#------------ test -----------------------------------------------

K = 2 # total number of iterates
Lx = 1 # smoothness constant along the block of coordinates x
Ly = 1 # smoothness constant along the block of coordinates y
obj =  pep_alternating_minimization(K,Lx,Ly)[1]

# ---------------------------------------------------------------------------
# PEP formulation for CACD
# ---------------------------------------------------------------------------
# Computes the convex formulation of the PEP for CACD
# See Kamri et al., "On the Worst-Case Analysis of Cyclic Block Coordinate
# Descent Type Algorithms" (2025) for theoretical background.
function pep_cacd(K, nblocks, L)
    # K: total number of steps for CACD
    # nblocks: number of blocks
    # L: vector of smoothness
    dimG = K + 2
    dimF = K + 1
    

    # represents the different sequences of CACD
    ybar = [zeros(dimG) for _ in 1:K, __ in 1:nblocks]
    xbar = [zeros(dimG) for _ in 1:1, __ in 1:nblocks]
    zbar = [zeros(dimG) for _ in 1:(K+1), __ in 1:nblocks]
    ubar = [zeros(dimG) for _ in 1:(K+1), __ in 1:nblocks]

    
    for j in 1:nblocks
        xbar[1, j] .= zeros(dimG)
    end

    theta = zeros(Float64, K)
    theta[1] = 1 / nblocks
    for i in 1:K-1
        th = theta[i]
        theta[i+1] = 0.5 * (sqrt(th^4 + 4 * th^2) - th^2)
    end

   
    for i in 1:nblocks
        zbar[1, i][1] = 1.0
    end

    # gbar: represent the gradients
    gbar = [zeros(dimG) for _ in 1:dimG, __ in 1:nblocks]
    for j in 1:dimG
        for i in 1:nblocks
            if j <= dimG - 1
                gbar[j, i][j + 1] = 1.0
            end
        end
    end

    # fbar represents the functional values
    fbar = hcat(Matrix{Float64}(I, dimF, dimF), zeros(dimF))

    # Updates of CACD
    for i in 1:K
        idx = mod(i, nblocks) + 1
        for j in 1:nblocks
            ubar[i + 1, j] = ubar[i, j]
            zbar[i + 1, j] = zbar[i, j]
        end
        cst = 1 / (nblocks * theta[i] * L[idx])
        t = -cst * gbar[i, idx]
        zbar[i + 1, idx] = zbar[i, idx] + t
        ubar[i + 1, idx] = ubar[i, idx] - ((1 - nblocks * theta[i]) / (theta[i]^2)) * t
    end

    for i in 1:K
        for j in 1:nblocks
            ybar[i, j] = (theta[i]^2) * ubar[i, j] + zbar[i, j]
        end
    end

    for j in 1:nblocks
        xbar[1, j] = (theta[K]^2) * ubar[K + 1, j] + zbar[K + 1, j]
    end

    abs_ = [zeros(dimG) for _ in 1:dimG, __ in 1:nblocks]
    for j in 1:nblocks
        abs_[dimG, j] .= zeros(dimG)
    end
    for i in 1:K
        for j in 1:nblocks
            abs_[i, j] = ybar[i, j]
        end
    end
    for j in 1:nblocks
        abs_[K + 1, j] = xbar[1, j]
    end

    # Model
    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Variables:
    # F: functional values f_i = F' * fbar[:,i] 
    # G: Gram matrices for each block of coordinates
    # |x^(i)_k|^2 = xbar[k, i]'  * G[i] * xbar[k, i]
    # |g^(i)_k|^2 = gbar[k, i]' * G[i] * gbar[k, i]
    # g^(i)_k * x^(i)_j = gbar[k, i]' * G[i] * xbar[j, i]
    G = [@variable(model, [1:dimG, 1:dimG], PSD) for _ in 1:nblocks]
    @variable(model, F[1:dimF])

    # Initial condition for CACD
    condinit = 0.0
    for j in 1:nblocks
        condinit += L[j] * (abs_[1, j]' * G[j] * abs_[1, j])
    end
    @constraint(model, condinit <= 1.0)

    # Interpolation constraints for coordinate-wise smooth convex functions
    for i in 1:dimG
        for j in 1:dimG
            if i != j
                fi = fbar[:, i]
                fj = fbar[:, j]

                cond = F' * (fj - fi)
                for k in 1:nblocks
                    cond += gbar[j, k]' * G[k] * (abs_[i, k] - abs_[j, k])
                end
                for t in 1:nblocks
                    condt = cond + (1 / (2 * L[t])) * ((gbar[i, t] - gbar[j, t])' * G[t] * (gbar[i, t] - gbar[j, t]))
                    @constraint(model, condt <= 0.0)
                end
            end
        end
    end

    # Objective: f(x_K) - f(x_*)
    @objective(model, Max, F' * fbar[:, dimF])

    optimize!(model)

    objective = objective_value(model)
    G_res = [value.(G[j]) for j in 1:nblocks]
    F_res = value.(F)

    return objective, G_res, F_res, abs_, gbar, fbar, ubar
end

#---------------- test -----------------------------------------------------

K = 2 # total number of steps for CACD
nblocks = 2 # number of blocks of coordinates
L = [1,1] # vector of smoothness constants
obj = pep_cacd(K, nblocks, L)[1]

# ---------------------------------------------------------------------------
# PEP formulation for CCD from previous work for comparaison
# ---------------------------------------------------------------------------
# Computes the convex formulation of the PEP for CCD from Abbaszadehpeivasti et al "Convergence rate analysis of randomized and cyclic
# coordinate descent for convex optimization through semidefinite programming" link: https://arxiv.org/abs/2212.12384
function pep_ccd_HDZ(K, nblocks, L, h)

    # K: total number of iterates for CCD
    # nblocks: number of blocks for CCD
    # L: vector od smoothness constants
    # h: step size
    Lg = sum(L)
    dimG = K + 2
    dimF = K + 1

    # xbar, gbar represents resêctively iterates and associated gradients
    xbar = [zeros(dimG) for _ in 1:(K+2), __ in 1:nblocks]
    gbar = [zeros(dimG) for _ in 1:(K+2), __ in 1:nblocks]

    for i in 1:nblocks
        xbar[1, i][1] = 1.0
    end
    for j in 1:(K+2), i in 1:nblocks
        if j <= K + 1
            gbar[j, i][j + 1] = 1.0
        end
    end

    fbar = hcat(Matrix(I, dimF, dimF), zeros(dimF, 1)) # represents functional values

    # CCD updates
    for i in 1:K
        idx = mod(i, nblocks) + 1
        for j in 1:nblocks
            if j == idx
                xbar[i + 1, j] = xbar[i, j] .- (h / L[j]) .* gbar[i, j]
            else
                xbar[i + 1, j] = xbar[i, j]
            end
        end
    end

    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Variables:
    # F: functional values f_i = F' * fbar[:,i] 
    # G: Gram matrices for each block of coordinates
    # |x^(i)_k|^2 = xbar[k, i]'  * G[i] * xbar[k, i]
    # |g^(i)_k|^2 = gbar[k, i]' * G[i] * gbar[k, i]
    # g^(i)_k * x^(i)_j = gbar[k, i]' * G[i] * xbar[j, i]
    G = [@variable(model, [1:dimG, 1:dimG], PSD) for _ in 1:nblocks]
    @variable(model, F[1:dimF])

    # Inital condition of Setting ALL. (see paper https://arxiv.org/abs/2507.16675) 
    for i = 1:1+K
        if mod(i,nblocks) == 1
            condinit = 0
            for j = 1:nblocks
                condinit = condinit + xbar[i,j]'*G[j]*xbar[i,j]
            end
            @constraint(model, condinit <= 1.0)
        end
    end

    # Interpolation constraints for smooth convex functions
    for i in 1:(K+2)
        for j in 1:(K+2)
            if i != j
                fi = fbar[:, i]
                fj = fbar[:, j]

                cond = F' * (fj - fi)
                for k in 1:nblocks
                    cond += gbar[j, k]' * G[k] * (xbar[i, k] - xbar[j, k])
                end

                for t in 1:nblocks
                    dgt = gbar[i, t] - gbar[j, t]
                    cond = cond + (1 / (2 * Lg)) * (dgt' * G[t] * dgt)
                end
                @constraint(model, cond <= 0.0)
            end
        end
    end


    # additional valid constraints on successive iterates of CCD see https://arxiv.org/abs/2212.12384

    for i in 1:K
        j = i+1
        fi = fbar[:,i]
        fj = fbar[:,i+1]
        k = mod(i,nblocks) + 1
        cond1 =  F'*(fj-fi) + gbar[j,k]'*G[k]*(xbar[i,k]-xbar[j,k]) + 1/2/L[k] * ( (gbar[i,k]-gbar[j,k])'*G[k]*(gbar[i,k]-gbar[j,k]) )
        cond2 =  F'*(fi-fj) + gbar[i,k]'*G[k]*(xbar[j,k]-xbar[i,k]) + 1/2/L[k] * ( (gbar[i,k]-gbar[j,k])'*G[k]*(gbar[i,k]-gbar[j,k]) )
        @constraint(model, cond1 <= 0.0)
        @constraint(model, cond2 <= 0.0)
    end


    # Objective: maximize f(x_K)-f(x_*)
    obj_vec = fbar[:, K + 1]
    @objective(model, Max, F' * obj_vec)

    optimize!(model)

    objective = objective_value(model)
    G_val = [value.(G[j]) for j in 1:nblocks]
    return objective, G_val
end

#-------------- test -----------------------------------------------------
K = 2 # total number of steps for CCD
nblocks = 2 # number of blocks of coordinates
L = [1,1] # vector of smoothness constants
h = 1 # step size
obj = pep_ccd_HDZ(K, nblocks, L, h)[1]

# ---------------------------------------------------------------------------
# PEP formulation for lower bound on CCD convergence
# ---------------------------------------------------------------------------
# Computes using PEP a lower bound on the convergence of CCD as described in 
# Kamri et al., "On the worst-case analysis of cyclic coordinate-wise algorithms on smooth convex functions" 
# ECC23
function pep_ccd_lb(K, nblocks, L, h)

    # K: total number of iterates for CCD
    # nblocks: number of blocks for CCD
    # L: vector of smoothness constants
    # h: step size

    Lg = minimum(L)
    dimG = K + 2
    dimF = K + 1

    # xbar, gbar represents resêctively iterates and associated gradients
    xbar = [zeros(dimG) for _ in 1:(K+2), __ in 1:nblocks]
    gbar = [zeros(dimG) for _ in 1:(K+2), __ in 1:nblocks]

    for i in 1:nblocks
        xbar[1, i][1] = 1.0
    end
    for j in 1:(K+2), i in 1:nblocks
        if j <= K + 1
            gbar[j, i][j + 1] = 1.0
        end
    end

    fbar = hcat(Matrix(I, dimF, dimF), zeros(dimF, 1)) # represents functional values

    # CCD updates
    for i in 1:K
        idx = mod(i, nblocks) + 1
        for j in 1:nblocks
            if j == idx
                xbar[i + 1, j] = xbar[i, j] .- (h / L[j]) .* gbar[i, j]
            else
                xbar[i + 1, j] = xbar[i, j]
            end
        end
    end

    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Variables:
    # F: functional values f_i = F' * fbar[:,i] 
    # G: Gram matrices for each block of coordinates
    # |x^(i)_k|^2 = xbar[k, i]'  * G[i] * xbar[k, i]
    # |g^(i)_k|^2 = gbar[k, i]' * G[i] * gbar[k, i]
    # g^(i)_k * x^(i)_j = gbar[k, i]' * G[i] * xbar[j, i]
    G = [@variable(model, [1:dimG, 1:dimG], PSD) for _ in 1:nblocks]
    @variable(model, F[1:dimF])

    # Inital condition of Setting ALL. (see paper https://arxiv.org/abs/2507.16675) 
    for i = 1:1+K
        if mod(i,nblocks) == 1
            condinit = 0
            for j = 1:nblocks
                condinit = condinit + xbar[i,j]'*G[j]*xbar[i,j]
            end
            @constraint(model, condinit <= 1.0)
        end
    end

    # Interpolation constraints for smooth convex functions
    for i in 1:(K+2)
        for j in 1:(K+2)
            if i != j
                fi = fbar[:, i]
                fj = fbar[:, j]

                cond = F' * (fj - fi)
                for k in 1:nblocks
                    cond += gbar[j, k]' * G[k] * (xbar[i, k] - xbar[j, k])
                end

                for t in 1:nblocks
                    dgt = gbar[i, t] - gbar[j, t]
                    cond = cond + (1 / (2 * Lg)) * (dgt' * G[t] * dgt)
                end
                @constraint(model, cond <= 0.0)
            end
        end
    end



    # Objective: maximize f(x_K)-f(x_*)
    obj_vec = fbar[:, K + 1]
    @objective(model, Max, F' * obj_vec)

    optimize!(model)

    objective = objective_value(model)
    G_val = [value.(G[j]) for j in 1:nblocks]
    return objective, G_val
end

#-------------- test -----------------------------------------------------
K = 2 # total number of steps for CCD
nblocks = 2 # number of blocks of coordinates
L = [1,1] # vector of smoothness constants
h = 1 # step size
obj = pep_ccd_lb(K, nblocks, L, h)[1]