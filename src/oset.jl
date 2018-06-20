const ASet{T} = Base.AbstractSet{T}

struct IdSet{T} <: ASet{T}
  dict::IdDict{T,Nothing}
  IdSet{T}() where T = new(ObjectIdDict())
end

Base.eltype(::IdSet{T}) where T = T

IdSet() = IdSet{Any}()

Base.push!(s::IdSet{T}, x::T) where T = (s.dict[x] = nothing; s)
Base.delete!(s::IdSet{T}, x::T) where T = (delete!(s.dict, x); s)
Base.in(x, s::IdSet) = haskey(s.dict, x)

(::Type{IdSet{T}})(xs) where T = push!(IdSet{T}(), xs...)

IdSet(xs) = IdSet{eltype(xs)}(xs)

Base.collect(s::IdSet) = collect(keys(s.dict))
Base.similar(s::IdSet, T::Type) = IdSet{T}()

@forward IdSet.dict Base.length
