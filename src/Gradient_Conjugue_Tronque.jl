@doc doc"""
Minimise le problème : ``min_{||s||< \delta_{k}} q_k(s) = s^{t}g + (1/2)s^{t}Hs``
                        pour la ``k^{ème}`` itération de l'algorithme des régions de confiance

# Syntaxe
```julia
sk = Gradient_Conjugue_Tronque(fk,gradfk,hessfk,option)
```

# Entrées :   
   * **gradfk**           : (Array{Float,1}) le gradient de la fonction f appliqué au point xk
   * **hessfk**           : (Array{Float,2}) la Hessienne de la fonction f appliqué au point xk
   * **options**          : (Array{Float,1})
      - **delta**    : le rayon de la région de confiance
      - **max_iter** : le nombre maximal d'iterations
      - **tol**      : la tolérance pour la condition d'arrêt sur le gradient


# Sorties:
   * **s** : (Array{Float,1}) le pas s qui approche la solution du problème : ``min_{||s||< \delta_{k}} q(s)``

# Exemple d'appel:
```julia
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
xk = [1; 0]
options = []
s = Gradient_Conjugue_Tronque(gradf(xk),hessf(xk),options)
```
"""
function Gradient_Conjugue_Tronque(gradf,hessf,options)

    "# Si option est vide on initialise les 3 paramètres par défaut"
    if options == []
        delta = 2
        max_iter = 100
        tol = 1e-6
    else
        delta = options[1]
        max_iter = options[2]
        tol = options[3]
    end

    n = length(gradf)
    s = zeros(n)
    g0 = gradf
    g = g0
    p = -g0
    H = hessf
    
    q(x) = transpose(g)*x + 0.5*transpose(x)*H*x

    for i in 1 : max_iter
        kappa = transpose(p)*H*p
        
        # Grandeurs neccessaires a la resolution de l'equation :
        # norm(s_j + sigma p_j)^2 = Delta^2
        A = norm(p)^2
        B = dot(s, p)
        C = norm(s)^2 - delta^2
        
        # Recherche de la solution qui minimise q(s_j + sigma p_j)
        if kappa <= 0
            if A == 0
                return s
            end
            sigma1 = (-B + sqrt(B^2 - A*C))/A
            sigma2 = (-B - sqrt(B^2 - A*C))/A
            if q(s + sigma2*p) >= q(s + sigma1*p)
                s += sigma1*p
            else
                s += sigma2*p
            end
            return s
        end
        
        alpha = transpose(g)*g/kappa
        
        # Recherche de la solution positive
        if norm(s + alpha*p) >= delta
            if A == 0
                return s
            end
            sigma1 = (-B + sqrt(B^2 - A*C))/A
            return s + sigma1*p
        end
        
        # Mise a jour des parametres
        s += alpha*p
        gpre = g
        g += alpha*H*p
        beta = (transpose(g)*g)/(transpose(gpre)*gpre)
        p = -g + beta*p

        if norm(g) <= tol*norm(g0)
            return s
        end
    end
end
