#> This is an example program which solves a weakly coupled Laplace equation using OpenCMISS calls.
#>
#> By Chris Bradley
#>
#================================================================================================================================
#  Symbol Definitions
#================================================================================================================================

LINEAR_LAGRANGE = 1
QUADRATIC_LAGRANGE = 2
CUBIC_LAGRANGE = 3
CUBIC_HERMITE = 4
LINEAR_SIMPLEX = 5
QUADRATIC_SIMPLEX = 6
CUBIC_SIMPLEX = 7

#================================================================================================================================
#  User changeable example parameters
#================================================================================================================================

height = 1.0
width = 2.0
length = 3.0

numberOfGlobalXElements = 2
numberOfGlobalYElements = 2
numberOfGlobalZElements = 0

interpolationType = LINEAR_LAGRANGE

#================================================================================================================================
#  Other parameters
#================================================================================================================================

setupOutput = True
progressDiagnostics = True
debugLevel = 3

linearMaximumIterations      = 100000000 #default: 100000
linearRelativeTolerance      = 1.0E-4    #default: 1.0E-05
linearAbsoluteTolerance      = 1.0E-4    #default: 1.0E-10
linearDivergenceTolerance    = 1.0E5     #default: 1.0E5
linearRestartValue           = 30        #default: 30

contextUserNumber = 1

coordinateSystem1UserNumber = 1
coordinateSystem2UserNumber = 2
coordinateSystemInterfaceUserNumber = 3

region1UserNumber = 1
region2UserNumber = 2

basis1UserNumber = 1
basis2UserNumber = 2
basisInterfaceUserNumber = 3
basisInterfaceMappingUserNumber = 4

generatedMesh1UserNumber = 1
generatedMesh2UserNumber = 2
generatedMeshInterfaceUserNumber = 3

mesh1UserNumber = 1
mesh2UserNumber = 2
meshInterfaceUserNumber = 3

decomposition1UserNumber = 1
decomposition2UserNumber = 2
decompositionInterfaceUserNumber = 3

decomposerUserNumber = 1

geometricField1UserNumber = 1
geometricField2UserNumber = 2
geometricFieldInterfaceUserNumber = 3

equationsSetField1UserNumber = 4
equationsSetField2UserNumber = 5

equationsSet1UserNumber = 1
equationsSet2UserNumber = 2

dependentField1UserNumber = 4
dependentField2UserNumber = 5

interfaceUserNumber = 1
interfaceConditionUserNumber = 2

lagrangeFieldUserNumber = 2

problemUserNumber = 1

#================================================================================================================================
#  Initialise OpenCMISS
#================================================================================================================================

# Import the libraries (OpenCMISS,python,numpy,scipy)
import numpy,csv,time,sys,os,pdb
from opencmiss.opencmiss import OpenCMISS_Python as oc

# Override with command line arguments if need be
if len(sys.argv) > 1:
    if len(sys.argv) > 5:
        sys.exit('Error: too many arguments- currently only accepting 4 options: numberXElements numberYElements numberZElements interpolationType')
    if int(sys.argv[1]) >= 0:
        numberOfGlobalXElements = int(sys.argv[1])
    else:
        sys.exit('Error: The specified numberXElements of ' + int(sys.argv[1]) + ' is invalid. The number should be >= 0')
    if len(sys.argv) > 2:
        if int(sys.argv[2]) >= 0:
            numberOfGlobalYElements = int(sys.argv[2])
        else:
            sys.exit('Error: The specified numberYElements of ' + int(sys.argv[2]) + ' is invalid. The number should be >= 0')
    if len(sys.argv) > 3:
        if int(sys.argv[3]) >= 0:
            numberOfGlobalZElements = int(sys.argv[3])
        else:
            sys.exit('Error: The specified numberZElements of ' + int(sys.argv[3]) + ' is invalid. The number should be >= 0')
    if len(sys.argv) > 4:
        if int(sys.argv[4]) == LINEAR_LAGRANGE:
            interpolationType = LINEAR_LAGRANGE
        elif int(sys.argv[4]) == QUADRATIC_LAGRANGE:
            interpolationType = QUADRATIC_LAGRANGE
        elif int(sys.argv[4]) == CUBIC_LAGRANGE:
            interpolationType = CUBIC_LAGRANGE
        elif int(sys.argv[4]) == CUBIC_HERMITE:
            interpolationType = CUBIC_HERMITE
        else:
            sys.exit('Error: The specified interpolationType of ' + int(sys.argv[3]) + ' is invalid.')
        
# Diagnostics
#DiagnosticsSetOn(oc.DiagnosticTypes.ALL,[1,2,3,4,5],"Diagnostics",[""])
# Error Handling
#ErrorHandlingModeSet(oc.ErrorHandlingModes.TRAP_ERROR)
# Output
oc.OutputSetOn("Testing")

context = oc.Context()
context.Create(contextUserNumber)

worldRegion = oc.Region()
context.WorldRegionGet(worldRegion)

# Get the computational nodes info
computationEnvironment = oc.ComputationEnvironment()
context.ComputationEnvironmentGet(computationEnvironment)

worldWorkGroup = oc.WorkGroup()
computationEnvironment.WorldWorkGroupGet(worldWorkGroup)
numberOfComputationalNodes = worldWorkGroup.NumberOfGroupNodesGet()
computationalNodeNumber = worldWorkGroup.GroupNodeNumberGet()
          
# (NONE/TIMING/MATRIX/ELEMENT_MATRIX/NODAL_MATRIX)
equationsSet1OutputType = oc.EquationsSetOutputTypes.PROGRESS
equationsSet2OutputType = oc.EquationsSetOutputTypes.PROGRESS
equations1OutputType = oc.EquationsOutputTypes.NONE
equations2OutputType = oc.EquationsOutputTypes.MATRIX
interfaceConditionOutputType = oc.InterfaceConditionOutputTypes.PROGRESS
interfaceEquationsOutputType = oc.EquationsOutputTypes.NONE
coupledSolverOutputType = oc.SolverOutputTypes.MONITOR

if (numberOfGlobalZElements == 0):
    numberOfDimensions = 2
    numberOfInterfaceDimensions = 1
else:
    numberOfDimensions = 3
    numberOfInterfaceDimensions = 2

if (interpolationType == LINEAR_LAGRANGE):
    numberOfNodesXi = 2
    numberOfGaussXi = 2
    simplex = False
elif (interpolationType == QUADRATIC_LAGRANGE):
    numberOfNodesXi = 3
    numberOfGaussXi = 3
    simplex = False
elif (interpolationType == CUBIC_LAGRANGE):
    numberOfNodesXi = 4
    numberOfGaussXi = 3
    simplex = False
elif (interpolationType == CUBIC_HERMITE):
    numberOfNodesXi = 2
    numberOfGaussXi = 3
    simplex = False
elif (interpolationType == LINEAR_SIMPLEX):
    numberOfNodesXi = 2
    gaussOrder = 2
    simplex = True
    simplexOrder = 1
elif (interpolationType == QUADRATIC_SIMPLEX):
    numberOfNodesXi = 3
    gaussOrder = 4
    simplex = True
    simplexOrder = 2
elif (interpolationType == CUBIC_SIMPLEX):
    numberOfNodesXi = 4
    gaussOrder = 5
    simplex = True
    simplexOrder = 3
else:
    print('ERROR: Invalid interpolation error')
    exit()
    
if (setupOutput):
    print('SUMMARY')
    print('=======')
    print(' ')
    if (interpolationType == LINEAR_LAGRANGE):
        print('    Interpolation type: LINEAR_LAGRANGE')
    elif (interpolationType == QUADRATIC_LAGRANGE):
        print('    Interpolation type: QUADRATIC_LAGRANGE')
    elif (interpolationType == CUBIC_LAGRANGE):
        print('    Interpolation type: CUBIC_LAGRANGE')
    elif (interpolationType == CUBIC_HERMITE):
        print('    Interpolation type: CUBIC_HERMITE')
    elif (interpolationType == LINEAR_SIMPLEX):
        print('    Interpolation type: LINEAR_SIMPLEX')
    elif (interpolationType == QUADRATIC_SIMPLEX):
        print('    Interpolation type: QUADRATIC_SIMPLEX')
    elif (interpolationType == CUBIC_SIMPLEX):
        print('    Interpolation type: CUBIC_SIMPLEX')
    else:
        print('ERROR: Invalid interpolation type')
        exit()            
    print(' ')
    print('    Height: {0:f}'.format(height))
    print('    Width : {0:f}'.format(width))
    print('    Length: {0:f}'.format(length))
    print(' ')
    print('    Number of X elements: {0:d}'.format(numberOfGlobalXElements))
    print('    Number of Y elements: {0:d}'.format(numberOfGlobalYElements))
    print('    Number of Z elements: {0:d}'.format(numberOfGlobalZElements))

#================================================================================================================================
#  Coordinate Systems
#================================================================================================================================

if (progressDiagnostics):
    print(' ')
    print('Coordinate systems ...')

if (progressDiagnostics):
    print('  Creating coordinate system 1 ...')
    
coordinateSystem1 = oc.CoordinateSystem()
coordinateSystem1.CreateStart(coordinateSystem1UserNumber,context)
coordinateSystem1.DimensionSet(numberOfDimensions)
coordinateSystem1.CreateFinish()

if (progressDiagnostics):
    print('  Creating coordinate system 2 ...')
    
coordinateSystem2 = oc.CoordinateSystem()
coordinateSystem2.CreateStart(coordinateSystem2UserNumber,context)
coordinateSystem2.DimensionSet(numberOfDimensions)
coordinateSystem2.CreateFinish()

if (progressDiagnostics):
    print('  Creating interface coordinate system ...')
    
interfaceCoordinateSystem = oc.CoordinateSystem()
interfaceCoordinateSystem.CreateStart(coordinateSystemInterfaceUserNumber,context)
interfaceCoordinateSystem.DimensionSet(numberOfDimensions)
interfaceCoordinateSystem.CreateFinish()

if (progressDiagnostics):
    print('Coordinate systems ... Done')
     
#================================================================================================================================
#  Regions
#================================================================================================================================
  
if (progressDiagnostics):
    print('Regions ...')
    
if (progressDiagnostics):
    print('  Creating region 1 ...')
    
region1 = oc.Region()
region1.CreateStart(region1UserNumber,worldRegion)
region1.LabelSet('Region1')
region1.CoordinateSystemSet(coordinateSystem1)
region1.CreateFinish()

if (progressDiagnostics):
    print('  Creating region 2 ...')
    
region2 = oc.Region()
region2.CreateStart(region2UserNumber,worldRegion)
region2.LabelSet('Region2')
region2.CoordinateSystemSet(coordinateSystem2)
region2.CreateFinish()

if (progressDiagnostics):
    print('Regions ... Done')
    
#================================================================================================================================
#  Bases
#================================================================================================================================

if (progressDiagnostics):
    print('Basis functions ...')
    
if (progressDiagnostics):
    print('  Creating basis 1 ...')
    
basis1 = oc.Basis()
basis1.CreateStart(basis1UserNumber,context)
basis1.NumberOfXiSet(numberOfDimensions)
if (simplex):
    basis1.TypeSet(oc.BasisTypes.SIMPLEX)
    if (interpolationType == LINEAR_SIMPLEX):
        basis1.InterpolationXiSet([oc.BasisInterpolationSpecifications.LINEAR_SIMPLEX]*numberOfDimensions)
    elif (interpolationType == QUADRATIC_SIMPLEX):
        basis1.InterpolationXiSet([oc.BasisInterpolationSpecifications.QUADRATIC_SIMPLEX]*numberOfDimensions)
    elif (interpolationType == CUBIC_SIMPLEX):
        basis1.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_SIMPLEX]*numberOfDimensions)
    else:
        print('Invalid interpolation type for simplex')
        exit()
    basis1.QuadratureOrderSet(gaussOrder)
else:
    basis1.TypeSet(oc.BasisTypes.LAGRANGE_HERMITE_TP)
    if (interpolationType == LINEAR_LAGRANGE):
        basis1.InterpolationXiSet([oc.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*numberOfDimensions)
    elif (interpolationType == QUADRATIC_LAGRANGE):
        basis1.InterpolationXiSet([oc.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]*numberOfDimensions)
    elif (interpolationType == CUBIC_LAGRANGE):
        basis1.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_LAGRANGE]*numberOfDimensions)
    elif (interpolationType == CUBIC_HERMITE):
        basis1.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_HERMITE]*numberOfDimensions)
    else:
        print('Invalid interpolation type for non simplex')
        exit()
    basis1.QuadratureNumberOfGaussXiSet([numberOfGaussXi]*numberOfDimensions)
basis1.CreateFinish()

if (progressDiagnostics):
    print('  Creating basis 2 ...')
    
basis2 = oc.Basis()
basis2.CreateStart(basis2UserNumber,context)
basis2.NumberOfXiSet(numberOfDimensions)
if (simplex):
    basis2.TypeSet(oc.BasisTypes.SIMPLEX)
    if (interpolationType == LINEAR_SIMPLEX):
        basis2.InterpolationXiSet([oc.BasisInterpolationSpecifications.LINEAR_SIMPLEX]*numberOfDimensions)
    elif (interpolationType == QUADRATIC_SIMPLEX):
        basis2.InterpolationXiSet([oc.BasisInterpolationSpecifications.QUADRATIC_SIMPLEX]*numberOfDimensions)
    elif (interpolationType == CUBIC_SIMPLEX):
        basis2.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_SIMPLEX]*numberOfDimensions)
    else:
        print('Invalid interpolation type for simplex')
        exit()
    basis2.QuadratureOrderSet(gaussOrder)
else:
    basis2.TypeSet(oc.BasisTypes.LAGRANGE_HERMITE_TP)
    if (interpolationType == LINEAR_LAGRANGE):
        basis2.InterpolationXiSet([oc.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*numberOfDimensions)
    elif (interpolationType == QUADRATIC_LAGRANGE):
        basis2.InterpolationXiSet([oc.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]*numberOfDimensions)
    elif (interpolationType == CUBIC_LAGRANGE):
        basis2.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_LAGRANGE]*numberOfDimensions)
    elif (interpolationType == CUBIC_HERMITE):
        basis2.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_HERMITE]*numberOfDimensions)
    else:
        print('Invalid interpolation type for non simplex')
        exit()
    basis2.QuadratureNumberOfGaussXiSet([numberOfGaussXi]*numberOfDimensions)
basis2.CreateFinish()

if (progressDiagnostics):
    print('Basis functions ... Done')
    
#================================================================================================================================
#  Generated meshes
#================================================================================================================================

if (progressDiagnostics):
    print('Generated meshes ...')
    
if (progressDiagnostics):
    print('  Creating generated mesh 1 ...')

generatedMesh1 = oc.GeneratedMesh()
generatedMesh1.CreateStart(generatedMesh1UserNumber,region1)
generatedMesh1.TypeSet(oc.GeneratedMeshTypes.REGULAR)
generatedMesh1.BasisSet([basis1])
if (numberOfDimensions == 2):
    generatedMesh1.ExtentSet([width,height])
    generatedMesh1.NumberOfElementsSet([numberOfGlobalXElements,numberOfGlobalYElements])
else:
    generatedMesh1.ExtentSet([width,height,length])
    generatedMesh1.NumberOfElementsSet([numberOfGlobalXElements,numberOfGlobalYElements,numberOfGlobalZElements])
mesh1 = oc.Mesh()
generatedMesh1.CreateFinish(mesh1UserNumber,mesh1)

if (progressDiagnostics):
    print('  Creating generated mesh 2 ...')

generatedMesh2 = oc.GeneratedMesh()
generatedMesh2.CreateStart(generatedMesh2UserNumber,region2)
generatedMesh2.TypeSet(oc.GeneratedMeshTypes.REGULAR)
generatedMesh2.BasisSet([basis2])
if (numberOfDimensions == 2):
    generatedMesh2.OriginSet([width,0.0])
    generatedMesh2.ExtentSet([width,height])
    generatedMesh2.NumberOfElementsSet([numberOfGlobalXElements,numberOfGlobalYElements])
else:
    generatedMesh2.OriginSet([width,0.0,0.0])
    generatedMesh2.ExtentSet([width,height,length])
    generatedMesh2.NumberOfElementsSet([numberOfGlobalXElements,numberOfGlobalYElements,numberOfGlobalZElements])
mesh2 = oc.Mesh()
generatedMesh2.CreateFinish(mesh2UserNumber,mesh2)

if (progressDiagnostics):
    print('Generated meshes ... Done')
    
#================================================================================================================================
#  Interface
#================================================================================================================================

if (progressDiagnostics):
    print('Interface ...')
    
if (progressDiagnostics):
    print('  Creating interface ...')
    
# Create an interface between the two meshes
interface = oc.Interface()
interface.CreateStart(interfaceUserNumber,worldRegion)
interface.LabelSet('Interface')
# Add in the two meshes
mesh1Index = interface.MeshAdd(mesh1)
mesh2Index = interface.MeshAdd(mesh2)
interface.CoordinateSystemSet(interfaceCoordinateSystem)
interface.CreateFinish()

if (progressDiagnostics):
    print('  Creating interface basis ...')
    
interfaceBasis = oc.Basis()
interfaceBasis.CreateStart(basisInterfaceUserNumber,context)
interfaceBasis.NumberOfXiSet(numberOfInterfaceDimensions)
if (simplex):
    interfaceBasis.TypeSet(oc.BasisTypes.SIMPLEX)
    if (interpolationType == LINEAR_SIMPLEX):
        interfaceBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.LINEAR_SIMPLEX]*numberOfInterfaceDimensions)
    elif (interpolationType == QUADRATIC_SIMPLEX):
        interfaceBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.QUADRATIC_SIMPLEX]*numberOfInterfaceDimensions)
    elif (interpolationType == CUBIC_SIMPLEX):
        interfaceBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_SIMPLEX]*numberOfInterfaceDimensions)
    else:
        print('Invalid interpolation type for simplex')
        exit()
    interfaceBasis.QuadratureOrderSet(gaussOrder)
else:
    interfaceBasis.TypeSet(oc.BasisTypes.LAGRANGE_HERMITE_TP)
    if (interpolationType == LINEAR_LAGRANGE):
        interfaceBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*numberOfInterfaceDimensions)
    elif (interpolationType == QUADRATIC_LAGRANGE):
        interfaceBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]*numberOfInterfaceDimensions)
    elif (interpolationType == CUBIC_LAGRANGE):
        interfaceBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_LAGRANGE]*numberOfInterfaceDimensions)
    elif (interpolationType == CUBIC_HERMITE):
        interfaceBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_HERMITE]*numberOfInterfaceDimensions)
    else:
        print('Invalid interpolation type for non simplex')
        exit()
    interfaceBasis.QuadratureNumberOfGaussXiSet([numberOfGaussXi]*numberOfInterfaceDimensions)
interfaceBasis.CreateFinish()

if (progressDiagnostics):
    print('  Creating interface mapping basis ...')
    
interfaceMappingBasis = oc.Basis()
interfaceMappingBasis.CreateStart(basisInterfaceMappingUserNumber,context)
interfaceMappingBasis.NumberOfXiSet(numberOfInterfaceDimensions)
if (simplex):
    interfaceMappingBasis.TypeSet(oc.BasisTypes.SIMPLEX)
    if (interpolationType == LINEAR_SIMPLEX):
        interfaceMappingBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.LINEAR_SIMPLEX]*numberOfInterfaceDimensions)
    elif (interpolationType == QUADRATIC_SIMPLEX):
        interfaceMappingBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.QUADRATIC_SIMPLEX]*numberOfInterfaceDimensions)
    elif (interpolationType == CUBIC_SIMPLEX):
        interfaceMappingBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_SIMPLEX]*numberOfInterfaceDimensions)
    else:
        print('Invalid interpolation type for simplex')
        exit()
    interfaceMappingBasis.QuadratureOrderSet(gaussOrder)
else:
    interfaceMappingBasis.TypeSet(oc.BasisTypes.LAGRANGE_HERMITE_TP)
    if (interpolationType == LINEAR_LAGRANGE):
        interfaceMappingBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*numberOfInterfaceDimensions)
    elif (interpolationType == QUADRATIC_LAGRANGE):
        interfaceMappingBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]*numberOfInterfaceDimensions)
    elif (interpolationType == CUBIC_LAGRANGE):
        interfaceMappingBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_LAGRANGE]*numberOfInterfaceDimensions)
    elif (interpolationType == CUBIC_HERMITE):
        interfaceMappingBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_HERMITE]*numberOfInterfaceDimensions)
    else:
        print('Invalid interpolation type for non simplex')
        exit()
    interfaceMappingBasis.QuadratureNumberOfGaussXiSet([numberOfGaussXi]*numberOfInterfaceDimensions)
interfaceMappingBasis.CreateFinish()

if (progressDiagnostics):
    print('  Creating interface generated mesh ...')

interfaceGeneratedMesh = oc.GeneratedMesh()
interfaceGeneratedMesh.CreateStartInterface(generatedMeshInterfaceUserNumber,interface)
interfaceGeneratedMesh.TypeSet(oc.GeneratedMeshTypes.REGULAR)
interfaceGeneratedMesh.BasisSet([interfaceBasis])
if (numberOfDimensions == 2):
    interfaceGeneratedMesh.OriginSet([width,0.0])
    interfaceGeneratedMesh.ExtentSet([0.0,height])
    interfaceGeneratedMesh.NumberOfElementsSet([numberOfGlobalYElements])
else:
    interfaceGeneratedMesh.OriginSet([width,0.0,0.0])
    interfaceGeneratedMesh.ExtentSet([0.0,height,length])
    interfaceGeneratedMesh.NumberOfElementsSet([numberOfGlobalYElements,numberOfGlobalZElements])
interfaceMesh = oc.Mesh()
interfaceGeneratedMesh.CreateFinish(meshInterfaceUserNumber,interfaceMesh)

if (progressDiagnostics):
    print('Interface ... Done')
    
#================================================================================================================================
#  Interface mesh connectivity
#================================================================================================================================

if (progressDiagnostics):
    print('Interface mesh connectivity ...')

# Couple the interface meshes
interfaceMeshConnectivity = oc.InterfaceMeshConnectivity()
interfaceMeshConnectivity.CreateStart(interface,interfaceMesh)
interfaceMeshConnectivity.BasisSet(interfaceBasis)
if (numberOfDimensions == 2):
    for yElementIdx in range(1,numberOfGlobalYElements+1):
        interfaceElementNumber = yElementIdx
        #Map the interface element to the elements in mesh 1
        mesh1ElementNumber = yElementIdx*numberOfGlobalXElements
        interfaceMeshConnectivity.ElementNumberSet(interfaceElementNumber,mesh1Index,mesh1ElementNumber)
        xi2 = [1.0,0.0]
        interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,mesh1Index,mesh1ElementNumber,1,1,xi2)
        xi2 = [1.0,1.0]
        interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,mesh1Index,mesh1ElementNumber,2,1,xi2)
        #Map the interface element to the elements in mesh 2
        mesh2ElementNumber = 1+(yElementIdx-1)*numberOfGlobalXElements
        interfaceMeshConnectivity.ElementNumberSet(interfaceElementNumber,mesh2Index,mesh2ElementNumber)
        for yLocalNodeIdx in range(1,numberOfNodesXi):
            xi2 = [0.0,(yLocalNodeIdx-1)/(numberOfNodesXi-1)]
            interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,mesh2Index,mesh2ElementNumber,1,1,xi2)
            xi2 = [0.0,yLocalNodeIdx/(numberOfNodesXi-1)]
            interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,mesh2Index,mesh2ElementNumber,2,1,xi2)
else:
    for yElementIdx in range(1,numberOfGlobalYElements+1):
        for zElementIdx in range(1,numberOfGlobalZElements+1):
            interfaceElementNumber = yElementIdx+(zElementIdx-1)*numberOfGlobalYElements
            #Map the interface element to the elements in mesh 1
            mesh1ElementNumber = yElementIdx*numberOfGlobalXElements+(zElementIdx-1)*numberOfGlobalXElements*numberOfGlobalYElements
            interfaceMeshConnectivity.ElementNumberSet(interfaceElementNumber,mesh1Index,mesh1ElementNumber)
            xi3 = [1.0,0.0,0.0]
            interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,mesh1Index,mesh1ElementNumber,1,1,xi3)
            xi3 = [1.0,1.0,0.0]
            interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,mesh1Index,mesh1ElementNumber,2,1,xi3)
            xi3 = [1.0,0.0,1.0]
            interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,mesh1Index,mesh1ElementNumber,3,1,xi3)
            xi3 = [1.0,1.0,1.0]
            interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,mesh1Index,mesh1ElementNumber,4,1,xi3)
            #Map the interface element to the elements in mesh 2
            mesh2ElementNumber = 1+(yElementIdx-1)*numberOfGlobalXElements+(zElementIdx-1)*numberOfGlobalXElements*numberOfGlobalYElements
            interfaceMeshConnectivity.ElementNumberSet(interfaceElementNumber,mesh2Index,mesh2ElementNumber)
            for yLocalNodeIdx in range(1,numberOfNodesXi):
                for zLocalNodeIdx in range(1,numberOfNodesXi):
                    xi3 = [0.0,(yLocalNodeIdx-1)/(numberOfNodesXi-1),(zLocalNodeIdx-1)/(numberOfNodesXi-1)]
                    interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,mesh2Index,mesh2ElementNumber,1,1,xi3)
                    xi3 = [0.0,yLocalNodeIdx/(numberOfNodesXi-1),(zLocalNodeIdx-1)/(numberOfNodesXi-1)]
                    interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,mesh2Index,mesh2ElementNumber,2,1,xi3)
                    xi3 = [0.0,(yLocalNodeIdx-1)/(numberOfNodesXi-1),zLocalNodeIdx/(numberOfNodesXi-1)]
                    interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,mesh2Index,mesh2ElementNumber,2,1,xi3)
                    xi3 = [0.0,yLocalNodeIdx/(numberOfNodesXi-1),zLocalNodeIdx/(numberOfNodesXi-1)]
                    interfaceMeshConnectivity.ElementXiSet(interfaceElementNumber,mesh2Index,mesh2ElementNumber,2,1,xi3)
interfaceMeshConnectivity.CreateFinish()

if (progressDiagnostics):
    print('Interface mesh connectivity ... Done')

#================================================================================================================================
#  Decomposition
#================================================================================================================================

if (progressDiagnostics):
    print('Decomposition ...')
              
if (progressDiagnostics):
    print('  Creating decomposition 1 ...')

# Create a decomposition for mesh 1
decomposition1 = oc.Decomposition()
decomposition1.CreateStart(decomposition1UserNumber,mesh1)
decomposition1.CalculateFacesSet(True)
decomposition1.CreateFinish()

if (progressDiagnostics):
    print('  Creating decomposition 2 ...')

# Create a decomposition for mesh 2
decomposition2 = oc.Decomposition()
decomposition2.CreateStart(decomposition2UserNumber,mesh2)
decomposition2.CalculateFacesSet(True)
decomposition2.CreateFinish()

if (progressDiagnostics):
    print('  Creating interface decomposition ...')

# Create a decomposition for interface mesh 
interfaceDecomposition = oc.Decomposition()
interfaceDecomposition.CreateStart(decompositionInterfaceUserNumber,interfaceMesh)
interfaceDecomposition.CreateFinish()

if (progressDiagnostics):
    print('Decomposition ... Done')
              
#================================================================================================================================
#  Decomposer
#================================================================================================================================

if (progressDiagnostics):
    print('Decomposer ...')
              
decomposer = oc.Decomposer()
decomposer.CreateStart(decomposerUserNumber,worldRegion,worldWorkGroup)
mesh1DecompositionIndex = decomposer.DecompositionAdd(decomposition1)
mesh2DecompositionIndex = decomposer.DecompositionAdd(decomposition2)
interfaceDecompositionIndex = decomposer.DecompositionAdd(interfaceDecomposition)
decomposer.OutputTypeSet(oc.DecomposerOutputTypes.ALL)    
decomposer.CreateFinish()
if (progressDiagnostics):
    print('Decomposer ... Done')
              
#================================================================================================================================
#  Geometric Field
#================================================================================================================================

if (progressDiagnostics):
    print('Geometric field ...')

if (progressDiagnostics):
    print('  Creating geometric field 1 ...')

# Start to create a default (geometric) field on region 1
geometricField1 = oc.Field()
geometricField1.CreateStart(geometricField1UserNumber,region1)
# Set the decomposition to use
geometricField1.DecompositionSet(decomposition1)
# Set the scaling to use
if (interpolationType == CUBIC_HERMITE):
    geometricField1.ScalingTypeSet(oc.FieldScalingTypes.ARITHMETIC_MEAN)
else:
    geometricField1.ScalingTypeSet(oc.FieldScalingTypes.NONE)
geometricField1.VariableLabelSet(oc.FieldVariableTypes.U,'Geometry1Variable')
# Set the domain to be used by the field components.
for componentIdx in range(1,numberOfDimensions+1):
    geometricField1.ComponentMeshComponentSet(oc.FieldVariableTypes.U,componentIdx,1)
# Finish creating the field
geometricField1.CreateFinish()

if (progressDiagnostics):
    print('  Creating geometric field 2 ...')

# Start to create a default (geometric) field on region 2
geometricField2 = oc.Field()
geometricField2.CreateStart(geometricField2UserNumber,region2)
# Set the decomposition to use
geometricField2.DecompositionSet(decomposition2)
# Set the scaling to use
if (interpolationType == CUBIC_HERMITE):
    geometricField2.ScalingTypeSet(oc.FieldScalingTypes.ARITHMETIC_MEAN)
else:
    geometricField2.ScalingTypeSet(oc.FieldScalingTypes.NONE)
geometricField2.VariableLabelSet(oc.FieldVariableTypes.U,'Geometry2Variable')
# Set the domain to be used by the field components.
for componentIdx in range(1,numberOfDimensions+1):
    geometricField2.ComponentMeshComponentSet(oc.FieldVariableTypes.U,componentIdx,1)
# Finish creating the field
geometricField2.CreateFinish()

if (progressDiagnostics):
    print('  Creating interface geometric field ...')

# Start to create a default (geometric) field on the interface
interfaceGeometricField = oc.Field()
interfaceGeometricField.CreateStartInterface(geometricFieldInterfaceUserNumber,interface)
# Set the decomposition to use
interfaceGeometricField.DecompositionSet(interfaceDecomposition)
# Set the scaling to use
if (interpolationType == CUBIC_HERMITE):
    interfaceGeometricField.ScalingTypeSet(oc.FieldScalingTypes.ARITHMETIC_MEAN)
else:
    interfaceGeometricField.ScalingTypeSet(oc.FieldScalingTypes.NONE)
interfaceGeometricField.VariableLabelSet(oc.FieldVariableTypes.U,'InterfaceGeometryVariable')
# Set the domain to be used by the field components.
for componentIdx in range(1,numberOfDimensions+1):
    interfaceGeometricField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,componentIdx,1)
# Finish creating the field
interfaceGeometricField.CreateFinish()

# Update the geometric field parameters
generatedMesh1.GeometricParametersCalculate(geometricField1)
generatedMesh2.GeometricParametersCalculate(geometricField2)
interfaceGeneratedMesh.GeometricParametersCalculate(interfaceGeometricField)

if (progressDiagnostics):
    print('Geometric field ... Done')
    
# Export the fields
fields1 = oc.Fields()
fields1.CreateRegion(region1)
fields1.NodesExport("CoupledLaplace1","FORTRAN")
fields1.ElementsExport("CoupledLaplace1","FORTRAN")

fields2 = oc.Fields()
fields2.CreateRegion(region2)
fields2.NodesExport("CoupledLaplace2","FORTRAN")
fields2.ElementsExport("CoupledLaplace2","FORTRAN")

interfaceFields = oc.Fields()
interfaceFields.CreateInterface(interface)
interfaceFields.NodesExport("CoupledLaplaceInterface","FORTRAN")
interfaceFields.ElementsExport("CoupledLaplaceInterface","FORTRAN")

#================================================================================================================================
#  Equations Set
#================================================================================================================================

if (progressDiagnostics):
    print('Equations sets ...')

if (progressDiagnostics):
    print('  Creating equations set 1 ...')

equationsSetField1 = oc.Field()
equationsSet1 = oc.EquationsSet()
equationsSet1Specification = [ oc.EquationsSetClasses.CLASSICAL_FIELD,
                               oc.EquationsSetTypes.LAPLACE_EQUATION,
                               oc.EquationsSetSubtypes.STANDARD_LAPLACE ]
equationsSet1.CreateStart(equationsSet1UserNumber,region1,geometricField1,equationsSet1Specification, \
                          equationsSetField1UserNumber,equationsSetField1)
equationsSet1.OutputTypeSet(equationsSet1OutputType)
equationsSet1.CreateFinish()

if (progressDiagnostics):
    print('  Creating equations set 2 ...')

equationsSetField2 = oc.Field()
equationsSet2 = oc.EquationsSet()
equationsSet2Specification = [ oc.EquationsSetClasses.CLASSICAL_FIELD,
                               oc.EquationsSetTypes.LAPLACE_EQUATION,
                               oc.EquationsSetSubtypes.STANDARD_LAPLACE ]
equationsSet2.CreateStart(equationsSet1UserNumber,region2,geometricField2,equationsSet2Specification, \
                          equationsSetField2UserNumber,equationsSetField2)
equationsSet2.OutputTypeSet(equationsSet1OutputType)
equationsSet2.CreateFinish()

if (progressDiagnostics):
    print('Equations sets ... Done')
    
#================================================================================================================================
#  Dependent fields
#================================================================================================================================

if (progressDiagnostics):
    print('Dependent fields ...')

if (progressDiagnostics):
    print('  Creating dependent field 1 ...')

dependentField1 = oc.Field()
equationsSet1.DependentCreateStart(dependentField1UserNumber,dependentField1)
equationsSet1.DependentCreateFinish()

if (progressDiagnostics):
    print('  Creating dependent field 2 ...')

dependentField2 = oc.Field()
equationsSet2.DependentCreateStart(dependentField2UserNumber,dependentField2)
equationsSet2.DependentCreateFinish()

if (progressDiagnostics):
    print('Dependent fields ... Done')
    
#================================================================================================================================
#  Equations
#================================================================================================================================

if (progressDiagnostics):
    print('Equations ...')

if (progressDiagnostics):
    print('  Creating eqations 1 ...')

equations1 = oc.Equations()
equationsSet1.EquationsCreateStart(equations1)
#equations1.SparsityTypeSet(oc.EquationsSparsityTypes.FULL)
equations1.SparsityTypeSet(oc.EquationsSparsityTypes.SPARSE)
equations1.OutputTypeSet(equations1OutputType)
equationsSet1.EquationsCreateFinish()

if (progressDiagnostics):
    print('  Creating eqations 2 ...')

equations2 = oc.Equations()
equationsSet2.EquationsCreateStart(equations2)
#equations2.SparsityTypeSet(oc.EquationsSparsityTypes.FULL)
equations2.SparsityTypeSet(oc.EquationsSparsityTypes.SPARSE)
equations2.OutputTypeSet(equations1OutputType)
equationsSet2.EquationsCreateFinish()

if (progressDiagnostics):
    print('Equations ... Done')

#================================================================================================================================
#  Interface Condition
#================================================================================================================================

if (progressDiagnostics):
    print('Interface Condition ...')
    
# Create an interface condition between the two equations sets
interfaceCondition = oc.InterfaceCondition()
interfaceCondition.CreateStart(interfaceConditionUserNumber,interface,interfaceGeometricField)
# Specify the method for the interface condition
interfaceCondition.MethodSet(oc.InterfaceConditionMethods.LAGRANGE_MULTIPLIERS)
# Specify the type of interface condition operator
interfaceCondition.OperatorSet(oc.InterfaceConditionOperators.FIELD_CONTINUITY)
# Add in the dependent variables from the equations sets
interfaceCondition.DependentVariableAdd(mesh1Index,equationsSet1,oc.FieldVariableTypes.U)
interfaceCondition.DependentVariableAdd(mesh2Index,equationsSet2,oc.FieldVariableTypes.U)
# Set the label
interfaceCondition.LabelSet("InterfaceCondition")
# Set the output type
interfaceCondition.OutputTypeSet(interfaceConditionOutputType)
# Finish creating the interface condition
interfaceCondition.CreateFinish()

if (progressDiagnostics):
    print('  Creating Lagrange field ...')

# Create the Lagrange multipliers field
interfaceLagrangeField = oc.Field()
interfaceCondition.LagrangeFieldCreateStart(lagrangeFieldUserNumber,interfaceLagrangeField)
interfaceLagrangeField.VariableLabelSet(oc.FieldVariableTypes.U,'InterfaceLagrange')
# Finish the Lagrange multipliers field
interfaceCondition.LagrangeFieldCreateFinish()
    
if (progressDiagnostics):
    print('  Creating interface equations ...')

# Create the interface condition equations
interfaceEquations = oc.InterfaceEquations()
interfaceCondition.EquationsCreateStart(interfaceEquations)
# Set the interface equations sparsity
#interfaceEquations.sparsityType = oc.EquationsSparsityTypes.FULL
interfaceEquations.sparsityType = oc.EquationsSparsityTypes.SPARSE
# Set the interface equations output
interfaceEquations.outputType = interfaceEquationsOutputType
# Finish creating the interface equations
interfaceCondition.EquationsCreateFinish()

if (progressDiagnostics):
    print('Interface condition ... Done')
    
#================================================================================================================================
#  Problem
#================================================================================================================================

if (progressDiagnostics):
    print('Problem ...')

# Create a problem
problem = oc.Problem()
problemSpecification = [ oc.ProblemClasses.CLASSICAL_FIELD,
                         oc.ProblemTypes.LAPLACE_EQUATION,
                         oc.ProblemSubtypes.STANDARD_LAPLACE ]
problem.CreateStart(problemUserNumber,context,problemSpecification)
problem.CreateFinish()

if (progressDiagnostics):
    print('Problems ... Done')

#================================================================================================================================
#  Control Loop
#================================================================================================================================

if (progressDiagnostics):
    print('Control Loops ...')

# Create the problem control loop
controlLoop = oc.ControlLoop()
problem.ControlLoopCreateStart()
problem.ControlLoopCreateFinish()

if (progressDiagnostics):
    print('Control Loops ... Done')

#================================================================================================================================
#  Solvers
#================================================================================================================================

if (progressDiagnostics):
    print('Solvers ...')

coupledSolver = oc.Solver()
problem.SolversCreateStart()
problem.SolverGet([oc.ControlLoopIdentifiers.NODE],1,coupledSolver)
coupledSolver.OutputTypeSet(coupledSolverOutputType)
coupledSolver.LinearTypeSet(oc.LinearSolverTypes.DIRECT)
coupledSolver.LibraryTypeSet(oc.SolverLibraries.MUMPS)
# Finish the creation of the problem solver
problem.SolversCreateFinish()

if (progressDiagnostics):
    print('Solvers ... Done')

#================================================================================================================================
#  Solver Equations
#================================================================================================================================

if (progressDiagnostics):
    print('Solver Equations ...')

# Start the creation of the problem solver equations
solverEquations = oc.SolverEquations()
problem.SolverEquationsCreateStart()
coupledSolver.SolverEquationsGet(solverEquations)
#solverEquations.SparsityTypeSet(oc.SolverEquationsSparsityTypes.FULL)
solverEquations.SparsityTypeSet(oc.SolverEquationsSparsityTypes.SPARSE)
solverEquations1Index = solverEquations.EquationsSetAdd(equationsSet1)
solverEquations2Index = solverEquations.EquationsSetAdd(equationsSet2)
interfaceConditionIndex = solverEquations.InterfaceConditionAdd(interfaceCondition)
# Finish the creation of the problem solver equations
problem.SolverEquationsCreateFinish()

if (progressDiagnostics):
    print('Solver Equations ... Done')

#================================================================================================================================
#  Boundary Conditions
#================================================================================================================================

if (progressDiagnostics):
    print('Boundary Conditions ...')
    
# Start the creation of the boundary conditions
boundaryConditions = oc.BoundaryConditions()
solverEquations.BoundaryConditionsCreateStart(boundaryConditions)
# Set the first node to 0.0
firstNodeNumber = 1
firstNodeDomain = decomposition1.NodeDomainGet(1,firstNodeNumber)
if (firstNodeDomain == computationalNodeNumber):
    boundaryConditions.SetNode(dependentField1,oc.FieldVariableTypes.U,1,1,firstNodeNumber,1,oc.BoundaryConditionsTypes.FIXED,0.0)
# Set the last node to 1.0
nodes2 = oc.Nodes()
region2.NodesGet(nodes2)
lastNodeNumber = nodes2.NumberOfNodesGet()
lastNodeDomain = decomposition2.NodeDomainGet(1,lastNodeNumber)
if (lastNodeDomain == computationalNodeNumber):
    boundaryConditions.SetNode(dependentField2,oc.FieldVariableTypes.U,1,1,lastNodeNumber,1,oc.BoundaryConditionsTypes.FIXED,1.0)
solverEquations.BoundaryConditionsCreateFinish()

if (progressDiagnostics):
    print('Boundary Conditions ... Done')
    
#================================================================================================================================
#  Run Solvers
#================================================================================================================================

# Solve the problem
if (progressDiagnostics):
    print('Solving problem...')
start = time.time()
problem.Solve()
end = time.time()
elapsed = end - start
print('Calculation Time = %3.4f' %elapsed)
if (progressDiagnostics):
    print('Problem solved!')

# Export the fields
fields1 = oc.Fields()
fields1.CreateRegion(region1)
fields1.NodesExport("CoupledLaplace1","FORTRAN")
fields1.ElementsExport("CoupledLaplace1","FORTRAN")

fields2 = oc.Fields()
fields2.CreateRegion(region2)
fields2.NodesExport("CoupledLaplace2","FORTRAN")
fields2.ElementsExport("CoupledLaplace2","FORTRAN")

interfaceFields = oc.Fields()
interfaceFields.CreateInterface(interface)
interfaceFields.NodesExport("CoupledLaplaceInterface","FORTRAN")
interfaceFields.ElementsExport("CoupledLaplaceInterface","FORTRAN")
