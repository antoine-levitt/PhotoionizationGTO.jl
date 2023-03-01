# m,n,o,p label AO
# i,j label occupied MO
# a,b label unoccupied MO
# μναβ label AO

using MKL
using LinearAlgebra
using FLoops
using PyCall
using Interpolations
using Tullio
using TimerOutputs
using IterativeSolvers
using LinearMaps
using PyPlot
using JLD2
using Folds
using FoldsThreads

#### Choose parameters

# atoms = "C 4.673795 6.280948 0.00; C 5.901190 5.572311 0.00; C 5.901190 4.155037 0.00; C 4.673795 3.446400 0.00; C 3.446400 4.155037 0.00; C 3.446400 5.572311 0.00; H 4.673795 7.376888 0.00; H 6.850301 6.120281 0.00; H 6.850301 3.607068 0.00; H 4.673795 2.350461 0.00; H 2.497289 3.607068 0.00; H 2.497289 6.120281 0.00" # benzene
# atoms = "F 0 0 0; H .917 0 0" # FH
# atoms = "Li 0 0 0; H 1.596 0 0" # LiH
# atoms = "H 0 0 0; H 0 0 .7414" # H2
# atoms = "O 0 0 0; H 0.7575 0.5871 0; H -0.7575 0.5871 0" # H2O
# atoms = "Li 0 0 0"
atoms = "He 0 0 0"
charge = 0


custom_grid = false
grid_level = 0
if custom_grid
    rad_grid_points = 100
    ang_grid_points = 6 # L=3
    pyimport("pyscf.dft")
    radi_method = pyscf.dft.radi.gauss_chebyshev
end
laug = (0, 1, )
naug = 0

zetas = ("X", "d", "t", "q", "5", "6")
keep_l = (0, 1, 2, 3, 4, 5, 6) # only keep these angular momenta in the basis
use_full_L = false # apply L iteratively, or build the full matrix
use_df = true # use density fitting to compute the Hartree potential on the grid
use_grid = true # compute G0 g on grid directly, or use P G0 g, where P is the basis
use_bare = false # bare response
use_free_G = false # use G0 instead of the true Green function. Useful to test convergence of z ψ0 wrt basis set
use_green = true # use the method (vs full ψ)
use_TDA = false # Tamm-Dancoff
use_interpolation = false
use_plot_casida = false
use_uncontracted = false
n_interp = 10
# η = .05 #.00001
η = 1e-10

nplot = 200
ωs = range(0, 2, nplot)


##### End choose parameters

BLAS.set_num_threads(Threads.nthreads())

## Normalizations: both pyscf and GaIn normalize (contracted) GTO in L^2
## χpyscf = pyscf_normalization*exp(-α r^2)*Ylm
pyscf_normalization(l, α) = pyscf.gto.gto_norm(l,α)/sqrt(4π)

pyscf = pyimport("pyscf")
include("GaIn.jl")


for (zeta_level, aug_level) in (
    (2, "aug-"),
    (3, "aug-"),
    # (4, "aug-"),
    # (5, "aug-"),
    # (2, "d-aug-"),
    # (3, "d-aug-"),
    # (4, "d-aug-"),
    # (5, "d-aug-"),
)
mol_temp = pyscf.gto.M(atom = atoms, basis = "STO-3G", charge=charge) # dummy
if mol_temp.natm == 1 && (mol_temp.atom_charges()[1] in (2, 3, 4))
    keep_l = (0, 1)
end

basis_label = "$(aug_level)cc-pv$(zetas[zeta_level])z.nw"
αmax = 10.0 * ones(length(laug))
αmin = .01 * ones(length(laug))
ntemp = naug * ones(Int, length(laug))
method = pyscf.scf.RKS
spdf = ["s", "p", "d", "f", "g"]
# aug_string = "+" * *(("$(ntemp[i])" * spdf[laug[i]+1] for i=1:length(laug))...)
aug_string = "+$naug"
naug == 0 && (aug_string="")


function augment_atomic_basis(atomic_basis, laug, αmin, αmax, ntemp, keep_l)
    atomic_basis = deepcopy(atomic_basis)
    atomic_basis = filter(b -> b[1] in keep_l, atomic_basis)
    for (il, l) in enumerate(laug)
        temp_exps = exp.(range(log(αmin[il]), log(αmax[il]), length=ntemp[il]))
        # temp_exps = (1/2) .^ (0:ntemp[il]-1)
        for α in temp_exps
            line = Any[l, [α, 1.0]]
            push!(atomic_basis, line)
        end
    end
    atomic_basis
end
dummy = pyscf.gto.M(atom = atoms, basis = "sto-3g", charge=charge) # just used to parse atom list
uncontract(x) = use_uncontracted ? collect(eachrow(pyscf.gto.uncontract(x))) : x
basis = Dict(elem => augment_atomic_basis(uncontract(pyscf.gto.basis.load(basis_label, elem)),
                                          laug, αmin, αmax, ntemp, keep_l)
             for elem in unique(dummy.atom_symbol.(0:(length(dummy.atom_charges())-1))))

mol = pyscf.gto.M(atom = atoms, basis = basis, charge=charge)
# plot_ref = mol.natm == 1 && (mol.atom_charges()[1] in (2, 4))

rhf = method(mol)
if custom_grid
    rhf.grids.radi_method = radi_method
    rhf.grids.atom_grid = (rad_grid_points, ang_grid_points)
else
    rhf.grids.level = grid_level
end
rhf.kernel()
basis_GaIn = build_αcrlm(mol)

# setup matrices
H = rhf.get_fock()
S = rhf.get_ovlp()
K = mol.intor("int1e_kin")
V = H-K
dirs = length(mol.atom_charges()) == 1 ? (1:1) : (1:3) # directions for polarizability
r_mat = mol.intor("int1e_r")[dirs, :, :]

# setup grid
grid_coords = rhf.grids.coords
weights = rhf.grids.weights
ao_value = pyscf.dft.numint.eval_ao(mol, grid_coords, deriv=0)

# function CAP(r)
#     rr = norm(r)
#     L = 1.0
#     η = .4
#     rr < L && return 0.0
#     -im*η*(rr-L)^2
# end
# pot_CAP = [CAP(r) for r in eachrow(grid_coords)]
# H = ComplexF64.(H)
# @tullio H[μ, ν] += ao_value[r, μ] * weights[r] * ao_value[r, ν] * pot_CAP[r]

# setup occupied, virtuals and rhs
n_tot = size(S, 1)
n_occ = mol.nelec[1]
n_virt = n_tot - n_occ
n_I = n_occ * n_virt
r_occ = 1:n_occ
r_virt = (n_occ+1):(n_occ+n_virt)
# eigdecomp = eigen(H, S)
# ψ = eigdecomp.vectors
# ε = eigdecomp.values
# ψ /= cholesky(Symmetric(ψ'*S*ψ)).L' # S-orthogonalize
# @assert norm(ψ'S*ψ - I) < 1e-10
# @assert norm(ψ*ψ'*S - rhf.mo_coeff*rhf.mo_coeff'*S) < 1e-10
ψ = rhf.mo_coeff
ε = rhf.mo_energy
ψocc = ψ[:, r_occ]
ψvirt = ψ[:, r_virt]
# rhs = <g, Vψ> (not rhs = ∑ rhs_i g_i)
rhs = map(dirs) do dir
    δVψ = r_mat[dir, :, :] * ψocc
    use_TDA ? δVψ : [δVψ;;; δVψ]
end

# setup value of things on the grid
@tullio ψ_value[r, i] := ao_value[r, μ] * ψocc[μ, i]
rho = pyscf.dft.numint.eval_rho(mol, ao_value, 2*ψocc*ψocc', xctype="LDA")
@assert norm(rho - 2sum(eachcol(ψ_value .^ 2))) < 1e-4
ex, vxc, fxc, kxc = pyscf.dft.libxc.eval_xc(rhf.xc, rho, deriv=2)
vxc = vxc[1]
fxc = fxc[1]
atom_charges = mol.atom_charges()
atom_coords = mol.atom_coords()
nuclear_potential(r, charges, coords) = sum(-charges[iat]/norm(r-coords[iat, :]) for iat=1:length(charges))


# Build val_hartree_codensities[r,μ,ν], Hartree potential on grid generated by codensities
function compute_val_hartree_codensities_df(grid_coords, basis_GaIn)
    if use_df
        auxmol = pyscf.df.addons.make_auxmol(mol)
        # (μν|P)
        int3c = pyscf.df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s1", comp=1)
        # (P|Q)
        int2c = auxmol.intor("int2c2e", aosym="s1", comp=1)

        basis_GaIn_aux = build_αcrlm(auxmol)
        val_hartree_aux = zeros(size(grid_coords, 1), length(basis_GaIn_aux.αs))
        @floop for ir = 1:size(grid_coords, 1)
            val_hartree_aux[ir, :] = build_vector((args...) -> value_coulomb_y(grid_coords[ir, :], args...), basis_GaIn_aux)
        end
        A = int2c \ transpose(val_hartree_aux) #Pr
        @tullio val_hartree_codensities[r, μ, ν] := int3c[μ, ν, P] * A[P, r]
    else
        val_hartree_codensities = zeros(size(grid_coords, 1), n_tot, n_tot)
        for ir = 1:size(grid_coords, 1)
            val_hartree_codensities[ir, :, :] = build_matrix((args...) -> value_coulomb_yy(grid_coords[ir, :], args...), basis_GaIn)
        end
        val_hartree_codensities
    end
end
println("basis set $(basis_label)$aug_string, grid_level $grid_level")
val_hartree_codensities = compute_val_hartree_codensities_df(grid_coords, basis_GaIn)
@tullio pot_hartree_grid[r] := val_hartree_codensities[r, μ, ν] * 2ψocc[μ, i] * ψocc[ν, i]
pot_nucl_grid = [nuclear_potential(r, atom_charges, atom_coords) for r in eachrow(grid_coords)]

V_grid = (pot_hartree_grid+pot_nucl_grid+vxc) .* weights
V_fromgrid = ao_value' * Diagonal(V_grid) * ao_value
print("Grid eig : ")
println(eigvals(K+V_fromgrid, S)[1:4])
print("Exact eig: ")
println(eigvals(K+V, S)[1:4])

# precompute <iμ|fhxc| on a grid
@tullio fhxcrμi[r, μ, i] := val_hartree_codensities[r, μ, ν]*ψocc[ν, i]
@tullio fhxcrμi[r, μ, i] += fxc[r] * ao_value[r, μ] * ψ_value[r, i]

function compute_G0(z, basis_GaIn=basis_GaIn)
    # yukawa gets us e^-k|x| / |x|. We want the Green function, solution of
    # (z + Δ/2)G = δ
    # (2z+Δ)G = 2δ
    # G = - e^(i sqrt(2z) |x|) / 2π |x|
    k = -im * sqrt(2z)
    build_matrix((args...) -> -y_yukawa_y(ComplexF64(k), args...) / (2π), basis_GaIn, ComplexF64)
end
function compute_G0_value(z)
    if use_grid
        val_G0 = zeros(ComplexF64, length(weights), length(basis_GaIn.αs))
        k = -im * sqrt(2z)
        @floop for ir = 1:size(grid_coords, 1)
            val_G0[ir, :] = build_vector((args...) -> -value_yukawa_y(grid_coords[ir, :], k, args...) / (2π), basis_GaIn, ComplexF64)
        end
        val_G0
    else
        G0 = S\compute_G0(z, basis_GaIn)
        ao_value * G0
    end
end


# Setup interpolation
if use_interpolation
    println("Precomputing G0...")
    begin
        max_E = ωs[end] + ε[n_occ]
        Es_interp = range(0, max_E, length=n_interp)
        G0s = [compute_G0(E+im*η) for E in Es_interp]
        G0_interp = LinearInterpolation(Es_interp, G0s)
        G0_values = [compute_G0_value(E+im*η) for E in Es_interp]
        G0_value_interp = LinearInterpolation(Es_interp, G0_values)
    end
end
get_G0(z) = use_interpolation ?  G0_interp(real(z)) : compute_G0(z)
get_G0_value(z) = use_interpolation ?  G0_value_interp(real(z)) : compute_G0_value(z)

# setup preconditionner
struct FunctionPreconditioner
    precondition
end
LinearAlgebra.ldiv!(y::T, P::FunctionPreconditioner, x) where {T} = (y .= P.precondition(x))
LinearAlgebra.ldiv!(P::FunctionPreconditioner, x) = (x .= P.precondition(x))

function α(ω; plot_stuff=false, use_project_virtuals=false)
    pm_max = use_TDA ? 1 : 2
    use_full_L && (L = zeros(ComplexF64, n_tot, n_occ, pm_max, n_tot, n_occ, pm_max))
    ϕtoψ = zeros(ComplexF64, n_tot, n_tot, n_occ, pm_max)
    use_ϕ = zeros(Bool, n_occ, pm_max)
    ao_or_G0ao_value = zeros(ComplexF64, length(weights), n_tot, n_occ, pm_max)
    VG0 = zeros(ComplexF64, n_tot, n_tot, n_occ, pm_max)
    E(j, pmj) = pmj == 1 ? ω+im*η+ε[j] : -ω+im*η+ε[j]
    for j = 1:n_occ
        for pmj = 1:pm_max
            Ej = E(j, pmj)
            use_ϕ[j, pmj] = use_green && real(Ej) > 0
            begin
                # diagonal terms: 1-VG0
                if use_ϕ[j, pmj]
                    ϕtoψ[:, :, j, pmj] = S \ get_G0(Ej)
                    ao_or_G0ao_value[:, :, j, pmj] = get_G0_value(Ej)
                    if !use_free_G
                        # @tullio threads=false VG0[μ, ν, $j, $pmj] = (pot_hartree_grid[r]+pot_nucl_grid[r]+vxc[r]) *
                        # weights[r]*ao_value[r, μ] * ao_or_G0ao_value[r, ν, $j, $pmj]
                        # BLAS doesn't like complex x real so we do it by hand. TODO remove after https://github.com/JuliaLang/julia/pull/43435
                        @views VG0[:, :, j, pmj] = transpose(ao_value) * (Diagonal(V_grid) * real(ao_or_G0ao_value[:, :, j, pmj]))
                        @views VG0[:, :, j, pmj] .+= im .* (transpose(ao_value) * (Diagonal(V_grid) * imag(ao_or_G0ao_value[:, :, j, pmj])))
                    end

                    use_full_L && (L[:, j, pmj, :, j, pmj] = S - VG0[:, :, j, pmj])
                else
                    ϕtoψ[:, :, j, pmj] = I(n_tot)
                    ao_or_G0ao_value[:, :, j, pmj] = ao_value

                    use_full_L && (L[:, j, pmj, :, j, pmj] = Ej*S - H)
                end
            end

            if use_full_L
                if !use_bare
                    # off-diagonal terms: Khxc G0
                    for i = 1:n_occ
                        @tullio threads=false A[μ, ν] := 2fhxcrμi[r, μ, i] * weights[r] * ao_or_G0ao_value[r, ν, j, pmj] * ψ_value[r, j]
                        for pmi = 1:pm_max
                            L[:, i, pmi, :, j, pmj] -= A
                        end
                    end
                end
            end
        end
    end


    if use_full_L
        function matricize(arr)
            l = length(arr)
            sl = Int(sqrt(l))
            reshape(arr, sl, sl)
        end
        L = matricize(L)
    end

    function project_virtuals(δϕ)
        # δϕ are coefficients in the AO basis in input and output
        if use_project_virtuals
            P(f) = ψvirt * ψvirt' * (S * f)
            mapslices(P, δϕ; dims=(1, ))
        else
            δϕ
        end
    end
    # "S" version of the previous one (when x is coefficients in the dual basis)
    project_virtuals_dual(x) = mapslices(f -> S*f, project_virtuals(mapslices(f -> S\f, x; dims=1)); dims=1)


    function apply_L(δϕ)
        Lδϕ = zero(δϕ)
        δϕ = project_virtuals(δϕ)
        δρ = zeros(ComplexF64, length(weights))
        @views for j = 1:n_occ
            for pmj = 1:pm_max
                if use_ϕ[j, pmj]
                    Lδϕ[:, j, pmj] = (S - VG0[:, :, j, pmj]) * δϕ[:, j, pmj]
                else
                    Lδϕ[:, j, pmj] = (E(j, pmj)*S-H) * δϕ[:, j, pmj]
                end
                δϕ_value = ao_or_G0ao_value[:, :, j, pmj] * δϕ[:, j, pmj]
                δρ .+= 2 .* ψ_value[:, j] .* δϕ_value
            end
        end
        @views if !use_bare
            # @tullio threads=false Lδϕ[μ, i, pmj] += - fhxcrμi[r, μ, i] * weights[r] * δρ[r]
            weighted_δρ = weights .* δρ
            weighted_δρ_real = real(weighted_δρ)
            weighted_δρ_imag = imag(weighted_δρ)
            for i = 1:n_occ
                for pmi = 1:pm_max
                    # BLAS doesn't like complex x real so we do it by hand. TODO remove after https://github.com/JuliaLang/julia/pull/43435
                    Lδϕ[:, i, pmi] .-= transpose(fhxcrμi[:, :, i]) * weighted_δρ_real
                    Lδϕ[:, i, pmi] .-= im .* (transpose(fhxcrμi[:, :, i]) * weighted_δρ_imag)
                end
            end
        end
        project_virtuals_dual(Lδϕ)
    end

    function apply_P(δϕ)
        Pδϕ = zero(δϕ)
        δϕ = project_virtuals_dual(δϕ)
        @views for j = 1:n_occ
            for pmj = 1:pm_max
                if use_ϕ[j, pmj]
                    # TODO regularize to avoid problems near resonances?
                    Pδϕ[:, j, pmj] = (S - VG0[:, :, j, pmj]) \ δϕ[:, j, pmj]
                else
                    Pδϕ[:, j, pmj] = (E(j, pmj)*S-H) \ δϕ[:, j, pmj]
                end
            end
        end
        project_virtuals(Pδϕ)
    end

    Lop = LinearMap{ComplexF64}(δϕ -> pack(apply_L(unpack(δϕ))), length(rhs[1]))
    Pop = FunctionPreconditioner(δϕ -> pack(apply_P(unpack(δϕ))))

    pack(ϕ) = vec(ϕ)
    unpack(ϕ) = reshape(ϕ, n_tot, n_occ, pm_max)

    αs = map(dirs) do dir
        if use_full_L
            δϕ = L \ rhs[dir]
        else
            δϕ, hist= gmres(Lop, pack(project_virtuals_dual(rhs[dir]));
                            log=true, Pl=Pop, abstol=1e-10, reltol=0.0, maxiter=60)
            δϕ = unpack(δϕ)
        end
        println("ω=$ω, gmres_its=$(hist.mvps), ionized=$(count(use_ϕ))")
        if plot_stuff
            begin
                j = pmj = 1
                δϕ_value = ao_or_G0ao_value[:, :, j, pmj] * δϕ[:, j, pmj]
                δρ = 2 .* ψ_value[:, j] .* δϕ_value
                mask = [i for i = 1:length(weights) if norm(grid_coords[i, 2:3]) == 0 && grid_coords[i, 1] ≥ 0]
                mask = mask[sortperm(grid_coords[mask, 1])]
                plot(grid_coords[mask, 1], real(ao_or_G0ao_value[mask, :, 1, 1] * δϕ[:, 1, 1]), label="Re(δψ)")
                # plot(grid_coords[mask, 1], imag(ao_or_G0ao_value[mask, :, 1, 1] * δϕ[:, 1, 1]), "-x", label="im δψ")
                plot(grid_coords[mask, 1], grid_coords[mask, 1] .^ 2 .* imag(δρ[mask]), label="Re(δϕ)")
                # plot(grid_coords[mask, 1], imag(ao_value[mask, :, 1, 1] * δϕ[:, 1, 1]), "-x", label="im δϕ")
                # plot(grid_coords[mask, 1], (ao_value[mask, :, 1, 1] * ψocc[:, 1]), "-x", label="ψ")
                # plot(grid_coords[mask, 1], (ao_value[mask, :, 1, 1] * (S \ rhs[1][:, 1, 1])), "-x", label="V ψ")
                legend()
            end
        end
        @tullio threads=false δψ[μ, i, pm] := ϕtoψ[μ, ν, i, pm] * δϕ[ν, i, pm]
        -2(dot(project_virtuals_dual(rhs[dir]), δψ))
    end
    αs
end
σ(ω) = 4π*ω/137.036*imag(mean(α(ω)))

# prop = pyimport("pyscf.prop.polarizability.rks")
# println("Static polarizability from us   : $(real.(α(1e-8im; use_project_virtuals=true)))")
# println("Static polarizability from pyscf: ", diag(rhf.Polarizability().polarizability()))

if use_plot_casida
    tdscf_rhf = pyimport("pyscf.tdscf.rhf")
    ab = tdscf_rhf.get_ab(rhf)
    A1 = reshape(ab[1], n_I, n_I)
    B1 = reshape(ab[2], n_I, n_I)
    C = [-A1 -B1; B1 A1]
    ωs_tddft = sort([λ for λ in real.(eigen(C).values) if λ > 0])
    # bare_spectrum = vcat((ε[n_occ+1:end] .- ε[i] for i=1:n_occ)...)
    plot(ωs_tddft, zero(ωs_tddft), "xk")
    xlim(ωs[1], ωs[end])
end

println("Computing σ...")
BLAS.set_num_threads(1)
σs = Folds.map(σ, ωs, FoldsThreads.WorkStealingEx())

filename = "res/$(atoms)/$(basis_label)$(aug_string).jld2"
mkpath(dirname(filename))
jldsave(filename; ωs, σs, ε, atoms, zeta_level, keep_l, grid_level, laug, naug, n_interp, use_interpolation, use_TDA, use_uncontracted, use_df, η)

PyPlot.rc("font", family="serif")
PyPlot.rc("xtick", labelsize="x-small")
PyPlot.rc("ytick", labelsize="x-small")
PyPlot.rc("figure", figsize=(4,3))
PyPlot.rc("text", usetex=false)

# aug_string = "+10"
label_noext = split(basis_label, ".")[1]
plot(ωs, σs, "-", label="$label_noext$aug_string")
xlabel("ω")
ylabel("σ")

for ip in ε[1:n_occ]
    axvline(-real(ip); c="k", ls="--")
end
legend()
xlim(ωs[1], ωs[end])

# if plot_ref
#     # Reference data from Karno
#     using DelimitedFiles
#     data = readdlm("$(mol.atom_symbol(0))_spectrum_sternheimer_tdlda.dat")
#     data = data[2:end, :]
#     plot(data[:, 1], data[:, 2], "-k")
#     xlim(ωs[1], ωs[end])
#     ylim(0, 1)
#     # ylim(-.2, 1)
# end

tight_layout()

savefig("res/$(atoms)/$(basis_label)$(aug_string).pdf")

# α(1; plot_stuff=true)
# xlabel("r")
# tight_layout()


end
