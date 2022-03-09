@doc doc"""
Minimise une fonction en utilisant l'algorithme des régions de confiance avec
    - le pas de Cauchy
ou
    - le pas issu de l'algorithme du gradient conjugue tronqué

# Syntaxe
```julia
xk, nb_iters, f(xk), flag = Regions_De_Confiance(algo,f,gradf,hessf,x0,option)
```

# Entrées :

   * **algo**        : (String) string indicant la méthode à utiliser pour calculer le pas
        - **"gct"**   : pour l'algorithme du gradient conjugué tronqué
        - **"cauchy"**: pour le pas de Cauchy
   * **f**           : (Function) la fonction à minimiser
   * **gradf**       : (Function) le gradient de la fonction f
   * **hessf**       : (Function) la hessiene de la fonction à minimiser
   * **x0**          : (Array{Float,1}) point de départ
   * **options**     : (Array{Float,1})
     * **deltaMax**      : utile pour les m-à-j de la région de confiance
                      ``R_{k}=\left\{x_{k}+s ;\|s\| \leq \Delta_{k}\right\}``
     * **gamma1,gamma2** : ``0 < \gamma_{1} < 1 < \gamma_{2}`` pour les m-à-j de ``R_{k}``
     * **eta1,eta2**     : ``0 < \eta_{1} < \eta_{2} < 1`` pour les m-à-j de ``R_{k}``
     * **delta0**        : le rayon de départ de la région de confiance
     * **max_iter**      : le nombre maximale d'iterations
     * **Tol_abs**       : la tolérence absolue
     * **Tol_rel**       : la tolérence relative

# Sorties:

   * **xmin**    : (Array{Float,1}) une approximation de la solution du problème : ``min_{x \in \mathbb{R}^{n}} f(x)``
   * **fxmin**   : (Float) ``f(x_{min})``
   * **flag**    : (Integer) un entier indiquant le critère sur lequel le programme à arrêter
      - **0**    : Convergence
      - **1**    : stagnation du ``x``
      - **2**    : stagnation du ``f``
      - **3**    : nombre maximal d'itération dépassé
   * **nb_iters** : (Integer)le nombre d'iteration qu'à fait le programme

# Exemple d'appel
```julia
algo="gct"
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
x0 = [1; 0]
options = []
xmin, fxmin, flag,nb_iters = Regions_De_Confiance(algo,f,gradf,hessf,x0,options)
```
"""

include("Pas_De_Cauchy.jl")

function Regions_De_Confiance(algo,f::Function,gradf::Function,hessf::Function,x0,options)

    if options == []
        deltaMax = 10
        gamma1 = 0.5
        gamma2 = 2.00
        eta1 = 0.25
        eta2 = 0.75
        delta0 = 2
        max_iter = 1000
        Tol_abs = sqrt(eps())
        Tol_rel = 1e-15
    else
        deltaMax = options[1]
        gamma1 = options[2]
        gamma2 = options[3]
        eta1 = options[4]
        eta2 = options[5]
        delta0 = options[6]
        max_iter = options[7]
        Tol_abs = options[8]
        Tol_rel = options[9]
    end

    n = length(x0)
    xmin = x0
    flag = -1
    nb_iters = 0
    delta = delta0
    normGradfx0 = norm(gradf(x0), 2)
    modifx = false
    e = 0
    
    enter = normGradfx0 > Tol_abs
    
    while flag == -1 && enter
        nb_iters += 1
        
        # Calcul de la solution du sous-probleme :
        # m_k(x_{k+1}) = min_{norm(x) <= Delta_k} m_k(x_k + s)
        if algo == "cauchy"
            s, e = Pas_De_Cauchy(gradf(xmin), hessf(xmin), delta)
        elseif algo == "gct"
            s = Gradient_Conjugue_Tronque(gradf(xmin), hessf(xmin), [delta max_iter Tol_abs])
        else
            throw("Mauvais argument\n--->Entrer 'cauchy' | 'gct'")
        end
      
        rho = (f(xmin) - f(xmin + s))/(-transpose(gradf(xmin))*s - 0.5*transpose(s)*hessf(xmin)*s)
        
        xpre = xmin
     
        # Mise a jour de l'itere courant et de la region de confiance
        if rho >= eta2
            xmin += s
            modifx = true
            delta = min(gamma2*delta, deltaMax)
        elseif rho >= eta1 && rho < eta2
            xmin += s
            modifx = true
        else
            delta *= gamma1
        end
        
        # Criteres de convergence
        if norm(gradf(xmin), 2) <= max(Tol_rel * normGradfx0, Tol_abs)
            flag = 0
        elseif norm(xmin - xpre, 2) <= max(Tol_rel * norm(xpre, 2), Tol_abs) && e == -1 && modifx
            flag = 1
        elseif abs(f(xmin) - f(xpre)) <= max(Tol_rel * abs(f(xpre)), Tol_abs) && e == -1 && modifx
            flag = 2
        elseif nb_iters >= max_iter
            flag = 3
        end
        
        modifx = false
    end
    
    fxmin = f(xmin)

    return xmin, fxmin, flag, nb_iters
end
