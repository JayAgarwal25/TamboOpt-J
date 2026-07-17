# tau_wholesky.jl
#
# Tau-neutrino injection -> tau propagation -> geometric cuts -> HDF5 dump.
# Chains the inject!/proposal_propagation! stages from
# examples/templates/{3_inject,4_propagate}.jl in a single process.
#
# Pipeline:
#   1. Inject nu_tau (pdg 16) over a zenith band around the horizon
#      (theta 60..100, phi 0..360) using the NeutrinoInjection strategy
#      (TauRunner Earth propagation + forced CC interaction at the detector
#      region).
#   2. Keep only events whose injection succeeded (have injection_final_state).
#   3. Propagate the resulting tau through rock+air with PROPOSAL.
#   4. Keep only events with a proposal_final_state (tau at end of track).
#   5. Cut events that are:
#        - still in the rock  -> !is_above_topography(final, terrain bvh)
#        - past the obs mesh  -> forward ray from the final state no longer
#                                crosses the detector/observation region
#                                (find_intersect(Ray(final), detector_bvh) === nothing),
#                                the same test the CORSIKA job planner uses.
#        - too close to the obs mesh -> decay vertex within --mindist metres of
#                                the observation surface (nearest point), so the
#                                shower has no room to develop.
#   6. Write the surviving taus' type, energy, direction, and position to HDF5.
#      Direction and position are expressed in the site-local ENU frame
#      (g_frame["cs"]: x=East, y=North, z=Up, origin at the site).
#
# Usage:
#   TAMBOSIM_PATH=/path/to/TamboSim julia tau_wholesky.jl [--nevent N] [--seed S]
#
# CLI flags (both `--flag value` and `--flag=value` forms accepted):
#   --nevent,  -n   number of events to inject   (default 50000)
#   --seed,    -s   RNG seed for injection+PROPOSAL (default: config value)
#   --outfile, -o   output HDF5 path (default: tau_wholesky.h5 next to this file)
#   --mindist       min decay-vertex -> obs-mesh distance, m (default 1000; 0 = off)

# TAMBOSIM_PATH locates the TamboSim checkout (this script lives in TambOpt, so
# @__DIR__ does not find it); its project also carries every dep used here.
tambo_path = get(ENV, "TAMBOSIM_PATH", dirname(@__DIR__))

using Pkg; Pkg.activate(tambo_path)

using TamboSim
using TOML
using HDF5
using LinearAlgebra: cross, dot, norm
using Unitful: ustrip, @u_str

# ---------------------------------------------------------------------------
# Minimal CLI parsing (avoids adding ArgParse to the scratch env)
# ---------------------------------------------------------------------------
function parse_args(argv)
    opts = Dict{String,Int}()
    aliases = Dict("-n" => "nevent", "-s" => "seed")
    i = 1
    while i <= length(argv)
        a = argv[i]
        key, val = if occursin('=', a)
            k, v = split(a, '=', limit=2); (k, v)
        else
            i += 1
            i <= length(argv) || error("missing value for argument $a")
            (a, argv[i])
        end
        key = lstrip(key, '-'); key = get(aliases, "-" * key, key)
        key in ("nevent", "seed", "mindist") || error("unknown argument: $a")
        opts[key] = parse(Int, val)
        i += 1
    end
    return opts
end

cli = parse_args(ARGS)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const GEOMETRY_FILE = joinpath(@__DIR__, "malata.jld2")
const CONFIG_FILE   = joinpath(tambo_path, "resources/configuration_examples/tau_neutrino_cc.toml")
const OUTFILE       = joinpath(@__DIR__, "tau_wholesky.h5")
const NEVENT        = get(cli, "nevent", 50_000)
const MINDIST       = get(cli, "mindist", 1000)   # m; 0 disables the cut

config = TOML.parsefile(CONFIG_FILE)
relativize!(config)

injection_config = config["injection"]
proposal_config  = config["proposal"]

# Seed: CLI overrides the config value for both injection and propagation
seed = get(cli, "seed", injection_config["seed"])
injection_config["seed"] = seed
proposal_config["seed"]  = seed

# nu_tau over a zenith band around the horizon. Was the whole sky (0..180),
# which wasted ~97% of throws: on the old 200k whole-sky corpus the survivors'
# directions all fell in zenith ~51..126 deg from +Up. 60..100 matches
# modules_v6/constants.py ZENITH_MIN/ZENITH_MAX. theta is measured from +Up on
# the direction of travel (sph_to_cart puts cos(theta) on the cs z-axis, and
# inject.jl samples `d` as the travel direction), so this is not mirrored.
injection_config["strategy"] = "NeutrinoInjection"
injection_config["pdg"]      = 16
injection_config["nevent"]   = NEVENT
injection_config["thetamin"] = 60    # deg
injection_config["thetamax"] = 100   # deg
injection_config["phimin"]   = 0     # deg
injection_config["phimax"]   = 360   # deg

println("nu_tau run")
println("  geometry : $GEOMETRY_FILE")
println("  nevent   : $NEVENT")
println("  seed     : $seed")
println("  zenith   : $(injection_config["thetamin"])..$(injection_config["thetamax"]) deg")
println("  azimuth  : $(injection_config["phimin"])..$(injection_config["phimax"]) deg")
println("  mindist  : $MINDIST m")

# ---------------------------------------------------------------------------
# 1-2. Inject and keep successful injections
# ---------------------------------------------------------------------------
frames = load_frames(GEOMETRY_FILE)
inject!(frames, injection_config)
filter!(f -> haskey(f, "injection_final_state"), frames)
println("After injection cut     : $(length(frames.q_frames)) Q frames")

# ---------------------------------------------------------------------------
# 3-4. Propagate the tau, keep events that produced a final state
# ---------------------------------------------------------------------------
proposal_propagation!(frames, proposal_config)
filter!(f -> haskey(f, "proposal_final_state"), frames)
println("After propagation       : $(length(frames.q_frames)) Q frames")

# ---------------------------------------------------------------------------
# 5. Geometric cuts: drop taus still in rock or past the observation mesh
# ---------------------------------------------------------------------------
g_frame      = frames.g_frames[end]
d_frame      = frames.d_frames[end]
bvh          = g_frame["bvh"]            # topography
detector_bvh = d_frame["detector_bvh"]   # observation / detector region
cs           = g_frame["cs"]             # site-local ENU frame

in_air(f)         = is_above_topography(f["proposal_final_state"], bvh)
before_obs_mesh(f) = !isnothing(TamboSim.find_intersect(Ray(f["proposal_final_state"]), detector_bvh))

filter!(f -> in_air(f) && before_obs_mesh(f), frames)
println("After in-rock/obs cuts  : $(length(frames.q_frames)) Q frames")

# ---------------------------------------------------------------------------
# 5b. Minimum decay-vertex -> observation-mesh distance.
# before_obs_mesh only asks whether the tau is still on the near side of the
# surface, so it keeps taus decaying ~0 m from it with no room to build a
# shower: on the old whole-sky corpus a quarter decayed within 26 m (p50 536 m).
# TamboSim has only ray/triangle intersection, so the point-to-mesh distance is
# computed here: perpendicular distance if the projection lands inside the face,
# else the nearest clamped edge.
# ---------------------------------------------------------------------------
function point_tri_dist(p, a, b, c)
    ab, ac = b .- a, c .- a
    n  = cross(ab, ac)
    nn = norm(n)
    nn < 1e-12 && return minimum(norm(p .- v) for v in (a, b, c))   # degenerate
    d = p .- a
    d00, d01, d11 = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    d20, d21 = dot(d, ab), dot(d, ac)
    den = d00 * d11 - d01 * d01
    if abs(den) > 1e-12
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        v >= 0 && w >= 0 && (v + w) <= 1 && return abs(dot(d, n ./ nn))
    end
    best = Inf
    for (p0, p1) in ((a, b), (b, c), (c, a))
        e  = p1 .- p0
        ee = dot(e, e)
        t  = ee < 1e-12 ? 0.0 : clamp(dot(p .- p0, e) / ee, 0.0, 1.0)
        best = min(best, norm(p .- (p0 .+ t .* e)))
    end
    return best
end

if MINDIST > 0
    obs_tris = [(ustrip.(u"m", convert(cs, t.v1).point),
                 ustrip.(u"m", convert(cs, t.v2).point),
                 ustrip.(u"m", convert(cs, t.v3).point)) for t in detector_bvh.triangles]
    far_enough(f) = minimum(
        point_tri_dist(ustrip.(u"m", convert(cs, f["proposal_final_state"].position).point), t...)
        for t in obs_tris) >= MINDIST
    filter!(far_enough, frames)
    println("After mindist cut       : $(length(frames.q_frames)) Q frames")
end

# ---------------------------------------------------------------------------
# 6. Collect fields and write HDF5
# ---------------------------------------------------------------------------
qf = frames.q_frames
n  = length(qf)

pdg       = Vector{Int}(undef, n)
energy    = Vector{Float64}(undef, n)      # GeV
direction = Matrix{Float64}(undef, n, 3)   # ENU unit vector
position  = Matrix{Float64}(undef, n, 3)   # ENU metres

for (i, f) in enumerate(qf)
    p = f["proposal_final_state"]
    pdg[i]         = Int(p.pdg)
    energy[i]      = ustrip(u"GeV", p.energy)
    position[i, :] = ustrip.(u"m", convert(cs, p.position).point)
    direction[i, :] = convert(cs, p.direction).point
end

mkpath(dirname(OUTFILE))
h5open(OUTFILE, "w") do file
    file["pdg"]       = pdg
    file["energy"]    = energy
    file["direction"] = direction
    file["position"]  = position
    attrs(file)["energy_units"]   = "GeV"
    attrs(file)["position_units"] = "m"
    attrs(file)["frame"]          = "site-local ENU (x=East, y=North, z=Up)"
    attrs(file)["direction_cols"] = "east north up"
    attrs(file)["position_cols"]  = "east north up"
    attrs(file)["nevent_thrown"]  = NEVENT
    attrs(file)["n_surviving"]    = n
    attrs(file)["thetamin_deg"]   = injection_config["thetamin"]
    attrs(file)["thetamax_deg"]   = injection_config["thetamax"]
    attrs(file)["mindist_m"]      = MINDIST
end

println("Saved $n events -> $OUTFILE")
