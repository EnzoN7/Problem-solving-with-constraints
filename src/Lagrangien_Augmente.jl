@doc doc"""
Résolution des problèmes de minimisation sous contraintes d'égalités

# Syntaxe
```julia
Lagrangien_Augmente(algo,fonc,contrainte,gradfonc,hessfonc,grad_contrainte,
			hess_contrainte,x0,options)
```

# Entrées
  * **algo** 		   : (String) l'algorithme sans contraintes à utiliser:
    - **"newton"**  : pour l'algorithme de Newton
    - **"cauchy"**  : pour le pas de Cauchy
    - **"gct"**     : pour le gradient conjugué tronqué
  * **fonc** 		   : (Function) la fonction à minimiser
  * **contrainte**	   : (Function) la contrainte [x est dans le domaine des contraintes ssi ``c(x)=0``]
  * **gradfonc**       : (Function) le gradient de la fonction
  * **hessfonc** 	   : (Function) la hessienne de la fonction
  * **grad_contrainte** : (Function) le gradient de la contrainte
  * **hess_contrainte** : (Function) la hessienne de la contrainte
  * **x0** 			   : (Array{Float,1}) la première composante du point de départ du Lagrangien
  * **options**		   : (Array{Float,1})
    1. **epsilon** 	   : utilisé dans les critères d'arrêt
    2. **tol**         : la tolérance utilisée dans les critères d'arrêt
    3. **itermax** 	   : nombre maximal d'itération dans la boucle principale
    4. **lambda0**	   : la deuxième composante du point de départ du Lagrangien
    5. **mu0,tho** 	   : valeurs initiales des variables de l'algorithme

# Sorties
* **xmin**		   : (Array{Float,1}) une approximation de la solution du problème avec contraintes
* **fxmin** 	   : (Float) ``f(x_{min})``
* **flag**		   : (Integer) indicateur du déroulement de l'algorithme
   - **0**    : convergence
   - **1**    : nombre maximal d'itération atteint
   - **(-1)** : une erreur s'est produite
* **niters** 	   : (Integer) nombre d'itérations réalisées

# Exemple d'appel
```julia
using LinearAlgebra
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
algo = "gct" # ou newton|gct
x0 = [1; 0]
options = []
contrainte(x) =  (x[1]^2) + (x[2]^2) -1.5
grad_contrainte(x) = [2*x[1] ;2*x[2]]
hess_contrainte(x) = [2 0;0 2]
output = Lagrangien_Augmente(algo,f,contrainte,gradf,hessf,grad_contrainte,hess_contrainte,x0,options)
```
"""
function Lagrangien_Augmente(algo,f::Function,c::Function,gradf::Function,
	hessf::Function,gradc::Function,hessc::Function,x0,options)

	if options == []
        tolAbs = 1e-8
        tolRel = 1e-5
        itermax = 1000
        lambda0 = 2
        mu0 = 100
        tau = 2
	else
        tolAbs = options[1]
        tolRel = options[2]
        itermax = options[3]
        lambda0 = options[4]
        mu0 = options[5]
        tau = options[6]
	end
    
    xmin = x0
    flag = -1
    iter = 0
    
    alpha = 0.1
    beta = 0.9

    eta0circ = 0.1258925
    epsilon0 = 1/mu0
    
    eta = eta0circ/mu0^alpha
    lambda = lambda0
    mu = mu0
    epsilon = epsilon0
    
    while flag == -1
        ### Definition du Lagrangien
        L(x) = f(x) + lambda'*c(x) + 0.5*mu*norm(c(x))^2
        gradL(x) = gradf(x) + lambda'*gradc(x) + mu*gradc(x)*c(x)
        hessL(x) = hessf(x) + lambda'*hessc(x) + mu*(hessc(x)*c(x) + gradc(x)*gradc(x)')
        
        ### Calcul de x_{k+1}
        if algo == "newton"
            xmin, _, _, _ = Algorithme_De_Newton(L, gradL, hessL, x0, [])
        elseif algo == "cauchy" || algo == "gct"
            xmin, _, _, _ = Regions_De_Confiance(algo, L, gradL, hessL, x0,  [])
        else
            throw("Mauvais argument\n--->Entrer 'newton' | 'cauchy' | 'gct'")
        end
        
        gradL_xmin_lambda_0 = gradf(xmin) + lambda'*gradc(xmin)
        gradL_x0_lambda0_0 = gradf(x0) + lambda0'*gradc(x0)
        
        ### Critere de convergence globale
        if norm(gradL_xmin_lambda_0) <= max(tolRel*norm(gradL_x0_lambda0_0), tolAbs) && norm(c(xmin)) <= max(tolRel*norm(c(x0)), tolAbs)
            flag = 0
        elseif iter >= itermax
            flag = 1
        end
        
        ### Mise a jour des multiplicateurs
        if norm(c(xmin)) <= eta
            lambda += mu*c(xmin)
            epsilon /= mu
            eta /= mu^beta
        else
            mu *= tau
            epsilon = epsilon0/mu
            eta = eta0circ/mu^alpha
        end
        iter += 1
    end
    
    fxmin = f(xmin)

    return xmin,fxmin,flag,iter
end
