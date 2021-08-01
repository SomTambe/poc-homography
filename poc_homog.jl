using Interpolations, 
      Zygote,
      ChainRulesCore,
      ChainRules,
      StaticArrays,
      ImageTransformations,
      Images,
      ImageProjectiveGeometry,
      LinearAlgebra,
      ColorVectorSpace,
      Plots

srcs = [[162/8,237/8,1] [220/8,979/8,1] [1237/8,861/8,1] [1254/8,301/8,1.]];
tgts = [[193/8,209/8,1] [224/8,854/8,1] [1308/8,867/8,1] [1359/8,227/8,1.]];
h = homography2d(srcs,tgts)

src_img = collect(load("/home/somvt/work/proof of concepts/source.jpeg")') .|> RGB{Float32}
tgt_img = collect(load("/home/somvt/work/proof of concepts/target.jpeg")') .|> RGB{Float32}

src2 = imresize(src_img, (100,100))
tgt2 = imresize(tgt_img, (100,100))

function ϕ(p, homo)
    q = inv(homo) * [p..., 1]
    q = q[1:2] ./ q[3]
    return q
end

function τ(X, q)
    y = interpolate(X, BSpline(Linear()))
    y = extrapolate(y, zero(eltype(X)))
    return y(q...)
end

function project(X, H)
    Y = similar(X)
    for i in CartesianIndices(Y)
        v = τ(X, ϕ(i.I, H))
        Y[i] = v
    end
    return Y
end

function ChainRulesCore.rrule(::typeof(τ), X, q)
    y = τ(X, q)
    function τ_pb(Δy)
        Δy = RGB(Δy)
        itp = extrapolate(interpolate(X, BSpline(Linear())), zero(eltype(X)))
        ∇X = NoTangent()
        ∇q = Interpolations.gradient(itp, q...)
        return NoTangent(), ∇X, typeof(q)(map(x->x.r*Δy.r+x.g*Δy.g+x.b*Δy.b, ∇q))
    end
    return y, τ_pb
end

function ChainRulesCore.rrule(::typeof(project), X, H)
    Y = project(X, H)
    function project_pb(Δy)
        ∇H = zeros(3,3)
        for p in CartesianIndices(Y)
            _, ∇τ = rrule(τ, X, Float32.(p.I))
            _, _, ∇τ = ∇τ(Δy[p])
            _, ∇ϕ = Zygote.pullback(ϕ, p.I, H)
            _, ∇h = ∇ϕ([∇τ...])
            ∇H += ∇h
        end
        return NoTangent(), NoTangent(), ∇H
    end
    return Y, project_pb
end

η = 2e-10
H = Matrix(1.0f0I, 3, 3)
outputs = []
for i in 1:100
    ∇H, = Zygote.gradient(H) do trfm
            project(src2, trfm) |> (z->sum(abs2, z - tgt2))
    end

    out = project(src2, H)
    println("Iteration: $i Loss: $(sum(abs2, out - tgt2))")

    H = H - η * ∇H
    push!(outputs, out')
end

@gif for i in 1:100
    plot(outputs[i], axis=([], false))
end