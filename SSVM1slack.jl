module SSVM1slack

using Convex, SCS

"""
Quadratic solver for 1-slack SVM
based on primal subgradient descent
"""
function pegasos(A, b, C, w0, maxiter=10000)
    d = length(A[1])
    w = w0
    for t = 1:maxiter
        subgradient = zeros(d)
        dis = [bi - dot(w, ai) for (ai, bi) in zip(A, b)]
        value, idx = findmax(dis)
        if value > 0.
            subgradient += A[idx]
        end
        eta_t = C/t
        w = (1 - 1/t) * w + eta_t * subgradient
    end
    dis = [bi - dot(w, ai) for (ai, bi) in zip(A, b)]
    value = maximum(dis)
    xi = max(0., value)
    (w, xi)
end

"""
Keep n first elements in list (used to purge cache)
If n=Inf, keep all elements.
"""
function head!(v::AbstractVector, n::Int)
    deleteat!(v, n+1:length(v))
end
function head!(v::AbstractVector, f::Float64)
    @assert f == Inf
    v
end

function addtocache!(cache, ais, bis)
    cached = cache[2]
    N = length(ais)
    for (i, a, b) in zip(1:N, ais, bis)
        push!(cached[i], (a, b))
    end
    cache
end

lencache(cache) = length(cache[2][1])

function planefromcache!(cache, w, n)
    cachesize, cached = cache
    ais = Vector{Float64}[]
    bis = Float64[]
    for i = 1:n
        c = cached[i]
        sort!(c, by=ab -> ab[2] - dot(ab[1], w), rev=true)
        head!(c, cachesize)
        push!(ais, c[1][1])
        push!(bis, c[1][2])
    end
    ais, bis
end

function batchinfer(maxOracle, X, w, verbose=1)
    N = length(X)
    Yhat = pmap(x -> maxOracle(w, x), X)
 #    if verbose >= 3
	# 	finished = [false for i = 1:N]
	# 	while !all(finished)
	# 		old_finished = finished
	# 		finished = [isready(r) for r in Yhat_remote]
	# 		for (i, f,o) in zip(1:N, finished, old_finished)
	# 			if f && !o
	# 				println("[Inference]: sample ", i, " finished.")
	# 			end
	# 		end
	# 		sleep(0.1)
	# 	end
	# end
 #    [fetch(r) for r in Yhat_remote]
end

function newconstraint{Ty}(psi, psigt, loss, maxOracle, X, Y::Vector{Ty}, w, verbose)
    N = length(X)
    # Yhat_remote = [@spawn maxOracle(w, X[i], Y[i]) for i = 1:N]
    # if verbose >= 3
    # 	finished = Bool[false for i = 1:N]
    # 	while !all(finished)
    # 		old_finished = finished
    # 		finished = Bool[isready(r) for r in Yhat_remote]
    # 		for (i, f,o) in zip(1:N, finished, old_finished)
    # 			if f && !o
    # 				println("[Augmented Inference]: sample ", i, " finished.")
    # 			end
    # 		end
    # 		sleep(0.1)
    # 	end
    # end
    # Yhat = Ty[fetch(r) for r in Yhat_remote]
    Yhat = pmap((x,y) -> maxOracle(w, x, y), X, Y)
    Yhat = Ty[yhat for yhat in Yhat] # assign correct type

    ais = Vector{Float64}[psigt[i] - psi(X[i], Yhat[i]) for i = 1:N]
    bis = Float64[loss(Y[i], Yhat[i]) for i = 1:N]
    (ais, bis)
end

const helpparam = Dict(
"X" => "Vector of training patterns",
"Y" => "Vector of training labels",
"featureFn" => "Joint feature function",
"lossFn" => "Loss function",
"oracleFn" => "Oracle function",
"c" => "SSVM regularization weight",
"positivity" => """index of additional positivity constraints in w, defaut Int[]
only implemented with quadratic == :primalqp""")

const helpoptions = Dict(
"quadratic" => """Quadratic optimization algorithm
either :pegasos (default, for sugradient descent) or :primalqp (for Convex.jl)""",
"eps" => "Constraint tolerance, default 0.1",
"num_passes" => "Number of iterations, default 200",
"max_time" => "Maximum running time in minutes, default Inf",
"cachesize" => "Size of cache per training example, default 10; can be 0 or Inf",
"mincachesize" => "Minimum size of the cache before we start using it, default 2",
"incremental" => """[10., 50.] will reduce the training set
to 10% in the first inference round,  50% in the second, and 100% in the next ones.""",
"Xtest" => "X_test list",
"Ytest" => "Y_test list",
"test_interval" => "Do the testing every ... iterations.",
"w0" => "initial w (defaults to zeros)",
"verbose" => "verbosity, default 1"
)


"""
Cutting-plane solver of Structural SVM in 1-slack formulation.
See helpparam or helpoptions for help on param and options

IN: param, options, helper (dictionary)
OUT: w, additional info in helper.

Implementation: Maxim Berman, 2015

Reference:
  T. Joachims, T. Finley, Chun-Nam Yu,
  Cutting-Plane Training of Structural SVMs, Machine Learning Journal.
"""
function ssvm_1slack(param, options, helper=Dict())
    time0 = time()
    
    # parse params
    X = param["X"]
    Y = param["Y"]    
    psi = param["featureFn"]
    loss = param["lossFn"]
    maxOracle = param["oracleFn"]
    N = length(X)
    c = get(param, "c", 0.01)
    positivity = get(param, "positivity", Int[])
    positivity_zero = get(param, "positivity_zero", 0.)

    # parse options
    max_iter = get(options, "num_passes", 200)
    max_time = 60 * get(options, "max_time", Inf)
    quadratic = get(options, "quadratic", :pegasos)
    eps = get(options, "eps", 0.1)
    cachesize = get(options, "cachesize", 10)
    mincachesize = get(options, "mincachesize", 2)
    incremental = get(options, "incremental", Float64[])
    Xtest = get(options, "Xtest", typeof(X)())
    Ytest = get(options, "Ytest", typeof(Y)())
    Ntest = length(Xtest)
    test_interval = get(options, "test_interval", 1)
    verbose = get(options, "verbose", 1)

    psigt = Vector{Float64}[psi(X[i], Y[i]) for i=1:N] # ground truth features
    jointdim = length(psigt[1])
    w = get(options, "w0", zeros(jointdim))
    
    # initializations
    N == length(Y) || error("# patterns != # labels.")
    Ntest == length(Ytest) || error("# test patterns != # test labels.")
    (
    length(positivity) == 0 || quadratic == :primalqp
    ) || error("positivity constraints only implemented in primalqp")
    
    xi = 0.

    A = Vector{Float64}[]  # lhs constraints
    b = Float64[]          # rhs consraints
    
    iter = 0
    converged = false
    
    helper["primal"] = Float64[]
    helper["violation"] = Float64[]
    helper["inferences"] = 0
    helper["inference_iter"] = Int[]
    helper["time_inference"] = Float64[]
    helper["time_quadratic"] = Float64[]
    helper["xi"] = Float64[]
    helper["test_loss"] = Float64[]
    helper["test_iter"] = Float64[]
    
    if cachesize > 0
        # cached[i][c] = c-th cached cutting plane tuple of example i
        cached = [Tuple{Vector{Float64}, Float64}[] for i=1:N]
        cache = (cachesize, cached)
    end
    
    C = c * N # rescale regularization by number of samples
    
    if quadratic == :primalqp
    	verbose >= 3 && println("[SSCM1slack] initializing Convex.jl problem")
        # initialize Convex.jl problem
        W = Variable(jointdim)
        W.value = w
        Xi = Variable(1, Positive())
        Xi.value = xi
        problem = minimize(0.5*sumsquares(W) + C*Xi)
        for idx in positivity
            problem.constraints += W[idx] >= positivity_zero
        end
    end
    
    n = 0 # number of partial training examples used
    inference_round = 0 # number of inference rounds
    
    test_iter = (Ntest > 0) ? test_interval : Inf # next testing iteration
    
    verbose >= 1 && println("[SSCM1slack] begin learning...")

    while (!converged)
        iter += 1

        verbose >= 2 && print("[SSCM1slack] Iter ", iter)

        converged = true
        
        if n < N
            # we haven't used the full training set yet,
            # so we can't stop here in any case
            converged = false
        end

        if lencache(cache) >= mincachesize
            # load constraints from cache
            ais, bis = planefromcache!(cache, w, n)
            ai, bi = mean(ais), mean(bis)
            violation = bi - dot(w, ai) - xi
            verbose >= 4 && print(" cache violation ", violation)
        end

        
        if lencache(cache) < mincachesize || violation < eps
            # cached constraints too weak, new inference...
            inference_round += 1
            
            oldn = n
            if inference_round <= length(incremental)
                npercent = incremental[inference_round]
                n = ceil(Int, N * npercent/100)
            else
                n = N
            end
            
            verbose >= 3 && print(": launching inference round ($n/$N samples)...\n")

            time1 = time()
            
            ais, bis = newconstraint(psi, psigt[1:n], loss, maxOracle, X[1:n], Y[1:n], w, verbose)
            
            # check coherence...
            for (ai, bi) in zip(ais, bis)
            	@assert bi - dot(ai, w) >= 0.
            end

            push!(helper["time_inference"], time() - time1)
            helper["inferences"] += n
            push!(helper["inference_iter"], iter)
            
            if 0 < oldn < n
                println("\n Augmenting training set")
                # we have added training examples,
                # so we must re-compute the means A and b in previous constraints
                newmeanA = mean(ais[oldn+1:n])
                newmeanb = mean(bis[oldn+1:n])
                for i = 1:length(A)
                    # reweighted mean
                    A[i][:] = (oldn * A[i][:] + (n - oldn) * newmeanA) / n
                    b[i] = (oldn * b[i] + (n - oldn) * newmeanb) / n
                end
                if quadratic == :primalqp
                    # rescale C in problem by the reduced number of samples
                    newproblem = minimize(0.5*sumsquares(W) + c*n*Xi)
                    for idx in positivity
                        newproblem.constraints += W[idx] >= 0.
                    end
                    solve!(newproblem, SCSSolver(verbose=false)) # workaround to create newproblem.model
                    newproblem.solution = problem.solution
                    newproblem.optval = problem.optval
                    problem = newproblem                         # forget old problem
                    for idx in positivity
                        problem.constraints += W[idx] >= positivity_zero
                    end
                    for (ai, bi) in zip(A, b)
                        problem.constraints += (dot(W, ai) >= bi - Xi)
                    end
                end
            end
            
            if cachesize > 0
                addtocache!(cache, ais, bis)
                emptycache = false
            end
            ai, bi = mean(ais), mean(bis)
            violation = bi - dot(w, ai) - xi
            verbose >= 4 && print("[SSCM1slack] New violation ", violation, "\n")
        else
            verbose >= 3 && print(": using cache...\n")
        end
        
        push!(helper["violation"], violation)
        if violation > eps
            converged = false
            
            time1 = time()
            
            push!(A, ai)
            push!(b, bi)
            
            verbose >= 3 && print("[SSCM1slack] Solving quadratic problem... ")
            if quadratic == :pegasos
                w, xi = pegasos(A, b, c * n, w)
            elseif quadratic == :primalqp
                problem.constraints += (dot(W, ai) >= bi - Xi)
                solve!(problem, SCSSolver(verbose=false), warmstart=true)
                w, xi = vec(evaluate(W)), evaluate(Xi)
            end
            w[positivity] = max(w[positivity], positivity_zero + zeros(length(positivity)))
            # show(w[positivity])
            verbose >= 3 && print("done!")
            push!(helper["time_quadratic"], time() - time1)
            
            push!(helper["xi"], xi)
            
            primal = 0.5*dot(w, w) + C*xi
            verbose >= 4 && print(" primal: $primal.\n")
            push!(helper["primal"], primal)
        end
        
        if (iter >= test_iter) # do the testing
            test_iter = iter + test_interval
            push!(helper["test_iter"], iter)
            verbose >= 3 && print("[SSCM1slack] Testing test set... ")
            Yhat = batchinfer(maxOracle, Xtest, w, verbose)
            testloss = mean([loss(Ytest[i], Yhat[i]) for i = 1:Ntest])
            verbose >= 3 && print("done! mean test loss: ", testloss, "\n")
            push!(helper["test_loss"], testloss)
        end
        
        if (n == N) && (iter >= max_iter)
            # we have considered the full training set and reached max_iter
            verbose >= 1 && print("[SSCM1slack] Finished!")
            break
        end

        if time() - time0 > max_time
        	verbose >= 1 && print("[SSCM1slack] Time limit exceeded.")
        	break
        end
       	
    end
    println()
    helper["inference_rounds"] = inference_round
    helper["iterations"] = iter
    helper["total_time"] = time() - time0
    w
end


end