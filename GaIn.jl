# to build GaIn as a shared library, get GaIn from Ivan and do
# ./configure FCFLAGS="-O2 -g -fPIC" CPPFLAGS="-O2 -g -fPIC"
# make
# cd src
# gfortran -shared -o lib.so *.o */*.o -lmpfr -lmpc

const GaIn_lib = expanduser("~/GaIn-1.0/src/lib.so")
struct Ccplx
    r::Cdouble
    i::Cdouble
end

# produces AO that are normalized in L^2 for the GaIn convention of integrals
function build_αcrlm(mol)
    αs = Vector{Float64}[] # α[ibas] is array of exponents
    cs = Vector{Float64}[] # c[ibas] is array of coefficients
    rs = Vector{Float64}[]
    ls = Int[]
    ms = Int[]
    for ibas = 0:mol.nbas-1
        l = convert(Int, mol.bas_angular(ibas))
        α = mol.bas_exp(ibas)[:]
        c_arr = mol.bas_ctr_coeff(ibas) # each column is an independent contraction
        iat = mol.bas_atom(ibas)
        r = mol.atom_coord(iat)

        for c in eachcol(c_arr)
            @assert length(α) == length(c)
            # each gaussian in the contraction is normalized in L^2
            c .*= α.^((3+2l)/4) # reverse engineered to make contractions work. Total scaling is off but relative is OK
            # now we renormalize to L^2, taking m=0 wlog
            total_L2_norm = sum(c1*c2*overlap(α1, r, l, 0, α2, r, l, 0) for (c1, α1) in zip(c, α), (c2, α2) in zip(c, α))
            c ./= sqrt(total_L2_norm)

            ml = -l:l
            if l == 1
                # pyscf uses the x,y,z ordering for the p orbitals, because fuck you...
                ml = (1, -1, 0)
            end
            for m in ml
                push!(ls, l)
                push!(rs, r)
                push!(ms, m)
                push!(αs, α)
                push!(cs, c)
            end
        end
    end
    (; αs, cs, rs, ls, ms)
end

function overlap(α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int, α2::Cdouble, r2::Vector{Cdouble}, l2::Int, m2::Int)
    @ccall GaIn_lib.y_overlap_y_(Ref(α1)::Ptr{Cdouble}, r1::Ptr{Cdouble}, Ref(Cint(l1))::Ptr{Cint}, Ref(Cint(m1))::Ptr{Cint},
                                 Ref(α2)::Ptr{Cdouble}, r2::Ptr{Cdouble}, Ref(Cint(l2))::Ptr{Cint}, Ref(Cint(m2))::Ptr{Cint})::Cdouble
end
function kinetic(α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int, α2::Cdouble, r2::Vector{Cdouble}, l2::Int, m2::Int)
    res =
        @ccall GaIn_lib.y_laplacian_y_(Ref(α1)::Ptr{Cdouble}, r1::Ptr{Cdouble}, Ref(Cint(l1))::Ptr{Cint}, Ref(Cint(m1))::Ptr{Cint},
                                       Ref(α2)::Ptr{Cdouble}, r2::Ptr{Cdouble}, Ref(Cint(l2))::Ptr{Cint}, Ref(Cint(m2))::Ptr{Cint})::Cdouble
    -res / 2
end
function y_yukawa_y(k::ComplexF64, α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int, α2::Cdouble, r2::Vector{Cdouble}, l2::Int, m2::Int)
    kvec = [k.re, k.im]
    res =
        @ccall GaIn_lib.y_yukawa_y_(kvec::Ptr{Cdouble},
                                     Ref(α1)::Ptr{Cdouble}, r1::Ptr{Cdouble}, Ref(Cint(l1))::Ptr{Cint}, Ref(Cint(m1))::Ptr{Cint},
                                     Ref(α2)::Ptr{Cdouble}, r2::Ptr{Cdouble}, Ref(Cint(l2))::Ptr{Cint}, Ref(Cint(m2))::Ptr{Cint})::Ccplx
    complex(res.r, res.i)
end
function yy_yukawa_y(k::ComplexF64,
                     α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int,
                     α2::Cdouble, r2::Vector{Cdouble}, l2::Int, m2::Int,
                     α3::Cdouble, r3::Vector{Cdouble}, l3::Int, m3::Int)
    kvec = [k.re, k.im]
    res = @ccall GaIn_lib.yy_yukawa_y_(kvec::Ptr{Cdouble},
                                        Ref(α1)::Ptr{Cdouble}, r1::Ptr{Cdouble}, Ref(Cint(l1))::Ptr{Cint}, Ref(Cint(m1))::Ptr{Cint},
                                        Ref(α2)::Ptr{Cdouble}, r2::Ptr{Cdouble}, Ref(Cint(l2))::Ptr{Cint}, Ref(Cint(m2))::Ptr{Cint},
                                        Ref(α3)::Ptr{Cdouble}, r3::Ptr{Cdouble}, Ref(Cint(l3))::Ptr{Cint}, Ref(Cint(m3))::Ptr{Cint})::Ccplx
    complex(res.r, res.i)
end
function ionic(α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int, α2::Cdouble, r2::Vector{Cdouble}, l2::Int, m2::Int, rion::Vector{Cdouble})
    res =
        @ccall GaIn_lib.yy_coulomb_ion_(Ref(α1)::Ptr{Cdouble}, r1::Ptr{Cdouble}, Ref(Cint(l1))::Ptr{Cint}, Ref(Cint(m1))::Ptr{Cint},
                                        Ref(α2)::Ptr{Cdouble}, r2::Ptr{Cdouble}, Ref(Cint(l2))::Ptr{Cint}, Ref(Cint(m2))::Ptr{Cint},
                                        rion::Ptr{Cdouble})::Cdouble
    -res
end

function yy_overlap_y(α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int,
                      α2::Cdouble, r2::Vector{Cdouble}, l2::Int, m2::Int,
                      α3::Cdouble, r3::Vector{Cdouble}, l3::Int, m3::Int)
    @ccall GaIn_lib.yy_overlap_y_(Ref(α1)::Ptr{Cdouble}, r1::Ptr{Cdouble}, Ref(Cint(l1))::Ptr{Cint}, Ref(Cint(m1))::Ptr{Cint},
                                  Ref(α2)::Ptr{Cdouble}, r2::Ptr{Cdouble}, Ref(Cint(l2))::Ptr{Cint}, Ref(Cint(m2))::Ptr{Cint},
                                  Ref(α3)::Ptr{Cdouble}, r3::Ptr{Cdouble}, Ref(Cint(l3))::Ptr{Cint}, Ref(Cint(m3))::Ptr{Cint}
                                  )::Cdouble
end

function yy_overlap_yy(α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int,
                       α2::Cdouble, r2::Vector{Cdouble}, l2::Int, m2::Int,
                       α3::Cdouble, r3::Vector{Cdouble}, l3::Int, m3::Int,
                       α4::Cdouble, r4::Vector{Cdouble}, l4::Int, m4::Int)
    @ccall GaIn_lib.yy_overlap_yy_(Ref(α1)::Ptr{Cdouble}, r1::Ptr{Cdouble}, Ref(Cint(l1))::Ptr{Cint}, Ref(Cint(m1))::Ptr{Cint},
                                   Ref(α2)::Ptr{Cdouble}, r2::Ptr{Cdouble}, Ref(Cint(l2))::Ptr{Cint}, Ref(Cint(m2))::Ptr{Cint},
                                   Ref(α3)::Ptr{Cdouble}, r3::Ptr{Cdouble}, Ref(Cint(l3))::Ptr{Cint}, Ref(Cint(m3))::Ptr{Cint},
                                   Ref(α4)::Ptr{Cdouble}, r4::Ptr{Cdouble}, Ref(Cint(l4))::Ptr{Cint}, Ref(Cint(m4))::Ptr{Cint}
                                   )::Cdouble
end
function yy_coulomb_yy(α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int,
                       α2::Cdouble, r2::Vector{Cdouble}, l2::Int, m2::Int,
                       α3::Cdouble, r3::Vector{Cdouble}, l3::Int, m3::Int,
                       α4::Cdouble, r4::Vector{Cdouble}, l4::Int, m4::Int)
    @ccall GaIn_lib.yy_coulomb_yy_(Ref(α1)::Ptr{Cdouble}, r1::Ptr{Cdouble}, Ref(Cint(l1))::Ptr{Cint}, Ref(Cint(m1))::Ptr{Cint},
                                   Ref(α2)::Ptr{Cdouble}, r2::Ptr{Cdouble}, Ref(Cint(l2))::Ptr{Cint}, Ref(Cint(m2))::Ptr{Cint},
                                   Ref(α3)::Ptr{Cdouble}, r3::Ptr{Cdouble}, Ref(Cint(l3))::Ptr{Cint}, Ref(Cint(m3))::Ptr{Cint},
                                   Ref(α4)::Ptr{Cdouble}, r4::Ptr{Cdouble}, Ref(Cint(l4))::Ptr{Cint}, Ref(Cint(m4))::Ptr{Cint}
                                   )::Cdouble
end
function yy_coulomb_y(α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int,
                      α2::Cdouble, r2::Vector{Cdouble}, l2::Int, m2::Int,
                      α3::Cdouble, r3::Vector{Cdouble}, l3::Int, m3::Int)
    @ccall GaIn_lib.yy_coulomb_y_(Ref(α1)::Ptr{Cdouble}, r1::Ptr{Cdouble}, Ref(Cint(l1))::Ptr{Cint}, Ref(Cint(m1))::Ptr{Cint},
                                  Ref(α2)::Ptr{Cdouble}, r2::Ptr{Cdouble}, Ref(Cint(l2))::Ptr{Cint}, Ref(Cint(m2))::Ptr{Cint},
                                  Ref(α3)::Ptr{Cdouble}, r3::Ptr{Cdouble}, Ref(Cint(l3))::Ptr{Cint}, Ref(Cint(m3))::Ptr{Cint}
                                  )::Cdouble
end
function y_coulomb_y(α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int,
                     α2::Cdouble, r2::Vector{Cdouble}, l2::Int, m2::Int)
    @ccall GaIn_lib.y_coulomb_y_(Ref(α1)::Ptr{Cdouble}, r1::Ptr{Cdouble}, Ref(Cint(l1))::Ptr{Cint}, Ref(Cint(m1))::Ptr{Cint},
                                 Ref(α2)::Ptr{Cdouble}, r2::Ptr{Cdouble}, Ref(Cint(l2))::Ptr{Cint}, Ref(Cint(m2))::Ptr{Cint}
                                 )::Cdouble
end

function value_coulomb_yy(rval::Vector{Cdouble},
                          α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int,
                          α2::Cdouble, r2::Vector{Cdouble}, l2::Int, m2::Int,)
    exp = 1e16
    charge_norm=2/π*(exp*sqrt(exp))
    charge_norm * yy_coulomb_y(α1, r1, l1, m1, α2, r2, l2, m2, exp, rval, 0, 0)
end
function value_coulomb_y(rval::Vector{Cdouble},
                         α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int)
    exp = 1e16
    charge_norm=2/π*(exp*sqrt(exp))
    charge_norm * y_coulomb_y(α1, r1, l1, m1, exp, rval, 0, 0)
end
function value_yukawa_y(rval::Vector{Cdouble}, k::ComplexF64, α1::Cdouble, r1::Vector{Cdouble}, l1::Int, m1::Int)
    exp = 1e16
    charge_norm=2/π*(exp*sqrt(exp))
    charge_norm * y_yukawa_y(k, α1, r1, l1, m1, exp, rval, 0, 0)
end

function build_vector(fun, basis_gain, T=Float64)
    (; αs, cs, rs, ls, ms) = basis_gain
    M = zeros(T, length(αs))
    for i = 1:length(αs)
        @assert length(αs[i]) == length(cs[i])
        for ic = 1:length(αs[i])
            M[i] += cs[i][ic] *
                fun(αs[i][ic], rs[i], ls[i], ms[i])
        end
    end
    M
end

function build_matrix(fun, basis_gain, T=Float64)
    (; αs, cs, rs, ls, ms) = basis_gain
    M = zeros(T, length(αs), length(αs))
    @floop for i = 1:length(αs)
        for j = 1:length(αs)
            local accum = zero(T) # keep it thread-local to minimize writes to M
            @assert length(αs[i]) == length(cs[i])
            @assert length(αs[j]) == length(cs[j])
            for ic = 1:length(αs[i])
                for jc = 1:length(αs[j])
                    accum += cs[i][ic] * cs[j][jc] *
                        fun(αs[i][ic], rs[i], ls[i], ms[i],
                            αs[j][jc], rs[j], ls[j], ms[j])
                end
            end
            M[i, j] = accum
        end
    end
    M
end

function build_4tensor(fun, basis_gain, T=Float64)
    (; αs, cs, rs, ls, ms) = basis_gain
    M = zeros(T, length(αs), length(αs), length(αs), length(αs))
    for i = 1:length(αs)
        for j = 1:length(αs)
            for k = 1:length(αs)
                for l = 1:length(αs)
                    @assert length(αs[i]) == length(cs[i])
                    @assert length(αs[j]) == length(cs[j])
                    for ic = 1:length(αs[i])
                        for jc = 1:length(αs[j])
                            for kc = 1:length(αs[k])
                                for lc = 1:length(αs[l])
                                    M[i, j, k, l] += cs[i][ic] * cs[j][jc] * cs[k][kc] * cs[l][lc] *
                                        fun(αs[i][ic], rs[i], ls[i], ms[i],
                                            αs[j][jc], rs[j], ls[j], ms[j],
                                            αs[k][kc], rs[k], ls[k], ms[k],
                                            αs[l][lc], rs[l], ls[l], ms[l]
                                            )
                                end
                            end
                        end
                    end
                    M
                end
            end
        end
    end
    M
end
