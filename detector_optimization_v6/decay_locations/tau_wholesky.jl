# tau_wholesky.jl
#
# Whole-sky tau-neutrino injection -> tau propagation -> geometric cuts ->
# HDF5 dump. Chains the inject!/proposal_propagation! stages from
# examples/templates/{3_inject,4_propagate}.jl in a single process.
#
# Pipeline:
#   1. Inject nu_tau (pdg 16) over the full sky (theta 0..180, phi 0..360)
#      using the NeutrinoInjection strategy (TauRunner Earth propagation +
#      forced CC interaction at the detector region).
#   2. Keep only events whose injection succeeded (have injection_final_state).
#   3. Propagate the resulting tau through rock+air with PROPOSAL.
#   4. Keep only events with a proposal_final_state (tau at end of track).
#   5. Cut events that are:
#        - still in the rock  -> !is_above_topography(final, terrain bvh)
#        - past the obs mesh  -> forward ray from the final state no longer
#                                crosses the detector/observation region
#                                (find_intersect(Ray(final), detector_bvh) === nothing),
#                                the same test the CORSIKA job planner uses.
#   6. Write the surviving taus' type, energy, direction, and position to HDF5.
#      Direction and position are expressed in the site-local ENU frame
#      (g_frame["cs"]: x=East, y=North, z=Up, origin at the site).
#
# Usage:
#   julia --project=scratch scratch/tau_wholesky.jl [--nevent N] [--seed S]
#
# CLI flags (both `--flag value` and `--flag=value` forms accepted):
#   --nevent, -n   number of events to inject   (default 50000)
#   --seed,   -s   RNG seed for injection+PROPOSAL (default: config value)

using Pkg; Pkg.activate(@__DIR__)

using TamboSim
using TOML
using HDF5
using Unitful: ustrip, @u_str

tambo_path = dirname(@__DIR__)

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
        key in ("nevent", "seed") || error("unknown argument: $a")
        opts[key] = parse(Int, val)
        i += 1
    end
    return opts
end

cli = parse_args(ARGS)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const GEOMETRY_FILE = joinpath(tambo_path, "resources/geometry/malata.jld2")
const CONFIG_FILE   = joinpath(tambo_path, "resources/configuration_examples/tau_neutrino_cc.toml")
const OUTFILE       = joinpath(@__DIR__, "tau_wholesky.h5")
const NEVENT        = get(cli, "nevent", 50_000)

config = TOML.parsefile(CONFIG_FILE)
relativize!(config)

injection_config = config["injection"]
proposal_config  = config["proposal"]

# Seed: CLI overrides the config value for both injection and propagation
seed = get(cli, "seed", injection_config["seed"])
injection_config["seed"] = seed
proposal_config["seed"]  = seed

# nu_tau over the whole sky
injection_config["strategy"] = "NeutrinoInjection"
injection_config["pdg"]      = 16
injection_config["nevent"]   = NEVENT
injection_config["thetamin"] = 0     # deg
injection_config["thetamax"] = 180   # deg
injection_config["phimin"]   = 0     # deg
injection_config["phimax"]   = 360   # deg

println("Whole-sky nu_tau run")
println("  geometry : $GEOMETRY_FILE")
println("  nevent   : $NEVENT")
println("  seed     : $seed")
println("  zenith   : $(injection_config["thetamin"])..$(injection_config["thetamax"]) deg")
println("  azimuth  : $(injection_config["phimin"])..$(injection_config["phimax"]) deg")

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
end

println("Saved $n events -> $OUTFILE")
