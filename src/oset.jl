const ASet{T} = Base.AbstractSet{T}
const ODict = ObjectIdDict

struct ObjectIdSet{T} <: ASet{T}
  dict::ObjectIdDict
  ObjectIdSet{T}() where T = new(ObjectIdDict())
end

Base.eltype{T}(::ObjectIdSet{T}) = T

ObjectIdSet() = ObjectIdSet{Any}()

Base.push!{T}(s::ObjectIdSet{T}, x::T) = (s.dict[x] = nothing; s)
Base.delete!{T}(s::ObjectIdSet{T}, x::T) = (delete!(s.dict, x); s)
Base.in(x, s::ObjectIdSet) = haskey(s.dict, x)

(::Type{ObjectIdSet{T}}){T}(xs) = push!(ObjectIdSet{T}(), xs...)

ObjectIdSet(xs) = ObjectIdSet{eltype(xs)}(xs)

Base.collect(s::ObjectIdSet) = collect(keys(s.dict))
Base.similar(s::ObjectIdSet, T::Type) = ObjectIdSet{T}()

@forward ObjectIdSet.dict Base.length

const OSet = ObjectIdSet
