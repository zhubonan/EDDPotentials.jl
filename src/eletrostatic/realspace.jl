# realspace summation
using LinearAlgebra
using SpecialFunctions: erfc, erf

"Compute volume of a lattice vector matrix (normal form)"
volume(cell::AbstractMatrix) = dot(cell[:, 1], cross(cell[:, 2], cell[:, 3]))

"""
  energy(lattice, positions, z, rc, rd)

Compute the electrostatic energy using real space summation technique
"""
function energy(lattice::AbstractMatrix, positions::AbstractMatrix, z::AbstractArray, 
                rc::Real, rd::Real)

    a1, a2, a3 = lattice[:, 1], lattice[:, 2], lattice[:, 3]
    vol = volume(lattice)
    rho_positive = sum(z[z .> 0]) / vol
    rho_negative = sum(z[z .< 0]) / vol

    nions = size(positions)[2]

    shift_vectors = compute_shifts(lattice, rc)
    nshifts = size(shift_vectors)[2]
    # Storage for pair-wise interactions/correction terms
    E_i = zeros((nions, nions))
    delta_E_i = zeros(nions)

    energy = 0.0   # Total energy
    for i in 1:nions
        ei = 0.0
        zi = z[i]
        qi_positive = zi
        qi_negative = zi

        for ishift in 1:nshifts
            @views origin_j = shift_vectors[:, ishift]
            for j in 1:nions
                # Skip for self interactions
                if (i == j) & (origin_j[1] == 0) & (origin_j[2] == 0) & (origin_j[3] == 0)
                    continue
                end

                dot_product = 0.0
                for icomp in 1:3
                    tmp = positions[icomp, i] - positions[icomp, j] - origin_j[icomp]
                    dot_product += tmp * tmp 
                end
                rij = sqrt(dot_product)
                if rij > rc
                    continue
                end

                zj = z[j]
                if zj > 0
                    qi_positive += zj
                elseif  zj < 0
                    qi_negative += zj
                end

                # Summation term
                E_i[i, j] += erfc(rij / rd) / rij * zj

            end # j
        end # ishift

        # apply 1/2 factors
        E_i[i, :] = E_i[i, :] * 0.5 * z[i]
        ei = sum(E_i[i, :])

        # Compute the correction term for positive and negative ions
        if rho_positive != 0.
            ra = compute_ra(qi_positive, rho_positive)
            delta_ei_positive = compute_delta_ei(ra, rho_positive, z[i], rd)
        else 
            delta_ei_positive = 0.0
        end
        if rho_negative != 0.
            ra = compute_ra(qi_negative, rho_negative)
            delta_ei_negative = compute_delta_ei(ra, rho_negative, z[i], rd)
        else 
            delta_ei_negative = 0.0
        end

        # Add everything up
        delta_ei = delta_ei_negative + delta_ei_positive
        delta_E_i[i] = delta_ei
        energy += ei + delta_ei
    end  # i
    return energy, E_i, delta_E_i
end

compute_ra(qi::Real, rho::Real) = abs(3.0 * qi / (4.0 * pi * rho)) ^ (1.0 / 3.0)

function  compute_delta_ei(ra::Real, rho::Real, zi::Real, rd::Real)
    sqrt_pi = sqrt(pi)
    sphere = -pi * zi * rho * ra * ra
    damp = pi * zi * rho * (ra * ra - rd * rd / 2.0) * erf(ra / rd) + sqrt_pi * zi * rho * ra * rd * exp(-ra * ra / (rd * rd))
    if zi * rho > 0.0
        self = -1.0 / (sqrt_pi * rd) * zi * zi
    else
        self = 0.0
    end
    return sphere + damp + self
end


"""
    compute_shifts(lattice, rc)

Compute the shift vectors needed to include all parts of the lattice within a
cut off radius
"""
function compute_shifts(lattice::AbstractMatrix, rc::Real)

    invl = inv(lattice)
    a1, a2, a3 = lattice[:, 1], lattice[:, 2], lattice[:, 3]
    b1, b2, b3 = invl[:, 1], invl[:, 2], invl[:, 3]

    shift1max = ceil(Int, rc * norm(b1))
    shift2max = ceil(Int, rc * norm(b2))
    shift3max = ceil(Int, rc * norm(b3))

    nshifts = (shift1max * 2 + 1) * (shift2max * 2 + 1) * (shift3max * 2 +1)

    shift_vectors = zeros(Float64, (3, nshifts))

    itmp = 1
    for shift3 in -shift3max:shift3max
        for shift2 in -shift2max:shift2max
            for shift1 in -shift1max:shift1max
                for i in 1:3
                    shift_vectors[i, itmp] = shift1 * a1[i] + shift2 * a2[i] + shift3 * a3[i]
                end
                itmp += 1
            end  # shift1
        end  # shift2
    end  # shift3

    return shift_vectors
end

