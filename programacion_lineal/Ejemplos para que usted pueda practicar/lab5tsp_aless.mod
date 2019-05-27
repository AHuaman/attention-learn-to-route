set nodo := 1..11;
set matriz := {nodo,nodo};
param distancia {matriz} >=0;
var x {matriz} binary;
var u {nodo} integer;

minimize Recorrido: sum{(i,j) in matriz: i!=j} distancia[i,j]*x[i,j];
# Llegan al punto de entrega
subject to Entrada {j in nodo}: sum{i in nodo: i!=j} x[i,j] = 1;
# Salen del punto de entrega
subject to Salida {i in nodo}: sum{j in nodo: i!=j} x[i,j] = 1;
# Solo se realiza un unico viaje
subject to Subtour {(i,j) in matriz: i!=j and i>1 and j>1}: u[i]-u[j]+11*x[i,j]<=10;
