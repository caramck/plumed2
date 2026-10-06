/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2026 of the visnet module authors (caramck).

The visnet module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The visnet module is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#ifdef __PLUMED_HAS_LIBTORCH

#include "core/Colvar.h"
#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include "tools/Units.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <dlfcn.h>

#include <cmath>
#include <fstream>
#include <map>
#include <regex>

namespace PLMD {
namespace visnet {

// When PLUMED is called from Python (e.g. ASE) and Python's torch is loaded in the same process,
// LibTorch uses the Python autograd engine, which refuses to run while the calling thread holds the
// GIL. PLUMED is not linked to Python, so the GIL functions are looked up at runtime: if Python is
// not loaded or this thread does not hold the GIL, this does nothing.
class ReleasePythonGIL {
  void (*restoreFrom)(void*) = nullptr;
  void* threadState = nullptr;
public:
  ReleasePythonGIL() {
    auto isInitialized=reinterpret_cast<int(*)()>(dlsym(RTLD_DEFAULT,"Py_IsInitialized"));
    auto holdsGIL=reinterpret_cast<int(*)()>(dlsym(RTLD_DEFAULT,"PyGILState_Check"));
    auto save=reinterpret_cast<void*(*)()>(dlsym(RTLD_DEFAULT,"PyEval_SaveThread"));
    restoreFrom=reinterpret_cast<void(*)(void*)>(dlsym(RTLD_DEFAULT,"PyEval_RestoreThread"));
    if(isInitialized && holdsGIL && save && restoreFrom && isInitialized() && holdsGIL()) {
      threadState=save();
    }
  }
  ~ReleasePythonGIL() {
    if(threadState) {
      restoreFrom(threadState);
    }
  }
  ReleasePythonGIL(const ReleasePythonGIL&) = delete;
  ReleasePythonGIL& operator=(const ReleasePythonGIL&) = delete;
};

//+PLUMEDOC COLVAR VISNET
/*
Use a ViSNet graph neural network, trained with torch_geometric, as a collective variable.

The model takes the atomic numbers and Cartesian positions of a group of atoms and returns a single
scalar (e.g. a HOMO-LUMO gap). The derivatives of the CV with respect to every atomic position are
computed at each step with the automatic differentiation of LibTorch, so the CV can be biased.
Since the model is not periodic, the virial is obtained from the positions and the atomic derivatives.
By default, the molecule is made whole across periodic boundaries before calling the model; use `NOPBC`
to disable this.

The order of the atoms does not matter, since ViSNet is invariant to permutations, but the i-th entry of
`TYPES` must be the atomic number of the i-th atom in `ATOMS`. A wrong pairing does not cause any error
in the model but silently changes the CV. For this reason, at the first step the atomic numbers are compared
with the masses provided by the MD code, and PLUMED stops if they do not match. If all masses are equal
to 1 or missing (as in `plumed driver` without `--mc` or `--pdb`) the check is skipped with a warning. Use `NOCHECK_TYPES` to
disable the check, e.g. with hydrogen mass repartitioning or isotope substitution.

The model must be exported to TorchScript with the `export_visnet.py` script shipped with the module,
which needs the trained weights (`model.state_dict()`) and a JSON "model card" with the
ViSNet hyperparameters, the target normalization and the length unit used for training.
The script replaces the `torch_cluster` neighbor search with a pure-torch one, folds the target
normalization into the model and checks the exported gradients against the original PyG model:

```bash
python export_visnet.py --weights weights.pt --card card.json --example frame.xyz -o model.pt
```

The positions are converted from the PLUMED length unit to the unit stored in the model card
(override it with `LENGTH_UNITS`). The CV is reported in the units of the trained target.

`PROPERTY` must name the property the model was trained or fine-tuned to predict, as stored in the
`target.name` field of the model card (e.g. `gap`); PLUMED stops if they differ, so that a model
trained on another property cannot be biased by mistake. The provenance stored in the card
(`level_of_theory`, `weights_file`, `finetuned_from`, `training_data`, `notes`) is written to the log.

Note that this action requires an installation of the LibTorch C++ library. Check the instructions on
[the module page](module_visnet.md) for how to enable the module.

## Examples

Compute the property predicted by a (random, test) model for an adenine molecule (atoms 1-15) and print it.
For a model fine-tuned on the HOMO-LUMO gap, `PROPERTY=gap` would be used instead.

```plumed
cv: VISNET ...
  ATOMS=1-15
  TYPES=1,1,1,1,1,6,6,6,6,6,7,7,7,7,7
  FILE=regtest/visnet/rt-visnet-derivatives/visnet_model.pt
  PROPERTY=test
...
PRINT ARG=cv FILE=COLVAR
```

*/
//+ENDPLUMEDOC

class VisnetModel : public Colvar {
  bool pbc;
  torch::jit::script::Module model;
  torch::Device device = torch::kCPU;
  torch::Dtype dtype = torch::kFloat32;
  torch::Tensor types;
  std::vector<int> atomicNumbers;
  bool checkTypes;
  double lengthScale;

  void checkTypesAgainstMasses();

  static std::string cardEntry(const std::string& card, const std::string& key);

public:
  explicit VisnetModel(const ActionOptions&);
  void calculate() override;
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(VisnetModel,"VISNET")

void VisnetModel::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);
  keys.add("atoms","ATOMS","the atoms that are passed to the model");
  keys.add("compulsory","TYPES","the atomic numbers of the atoms, in the same order as ATOMS");
  keys.add("compulsory","FILE","the TorchScript model created with export_visnet.py");
  keys.add("compulsory","PROPERTY","the property the model was trained to predict (e.g. gap). It must match the target name stored in the model card, so that the CV being biased is the intended one");
  keys.add("optional","LENGTH_UNITS","the length unit the model was trained with (A, nm, Bohr or a number in nm). By default it is read from the model card");
  keys.add("compulsory","DEVICE","cpu","the torch device to run the model on (cpu or cuda)");
  keys.addFlag("NOCHECK_TYPES",false,"do not check at the first step that TYPES is consistent with the masses of the atoms");
  keys.setValueDescription("scalar","the property predicted by the ViSNet model");
}

// Extract the string or number following "key": in the JSON model card
std::string VisnetModel::cardEntry(const std::string& card, const std::string& key) {
  std::smatch m;
  const std::regex re("\"" + key + "\"\\s*:\\s*(\"([^\"]*)\"|[-+0-9.eE]+)");
  if(!std::regex_search(card, m, re)) {
    return "";
  }
  return m[2].matched ? m[2].str() : m[1].str();
}

VisnetModel::VisnetModel(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true),
  checkTypes(true) {
  std::vector<AtomNumber> atoms;
  parseAtomList("ATOMS",atoms);
  if(atoms.empty()) {
    error("no atoms specified with ATOMS");
  }
  std::vector<int> z;
  parseVector("TYPES",z);
  if(z.size()!=atoms.size()) {
    error("TYPES should contain one atomic number for each of the " + std::to_string(atoms.size()) + " ATOMS");
  }
  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;
  bool nocheck=!checkTypes;
  parseFlag("NOCHECK_TYPES",nocheck);
  checkTypes=!nocheck;
  atomicNumbers=z;

  std::string fname;
  parse("FILE",fname);
  std::string devname;
  parse("DEVICE",devname);
  if(devname=="cuda") {
    if(!torch::cuda::is_available()) {
      error("DEVICE=cuda requested but CUDA is not available to LibTorch");
    }
    device=torch::kCUDA;
  } else if(devname!="cpu") {
    error("DEVICE should be cpu or cuda");
  }

  torch::jit::ExtraFilesMap extra_files{{"card.json",""}};
  try {
    model = torch::jit::load(fname, device, extra_files);
  } catch (const c10::Error& e) {
    std::ifstream infile(fname);
    if(infile.good()) {
      error("Cannot load FILE: '"+fname+"'. Please check that it was exported with export_visnet.py");
    } else {
      error("The FILE: '"+fname+"' does not exist.");
    }
  }
  model.eval();
  const std::string& card=extra_files["card.json"];
  if(card.empty()) {
    error("FILE '"+fname+"' has no model card: export it with export_visnet.py");
  }

  // the user must state which property is being used, and it must be the one the model was trained on
  std::string property;
  parse("PROPERTY",property);
  const std::string trainedProperty=cardEntry(card,"name");
  if(property!=trainedProperty) {
    error("PROPERTY="+property+" but the model in '"+fname+"' was trained to predict '"+trainedProperty+"'");
  }

  // the positions must be converted to the length unit used for training
  std::string modelUnits=cardEntry(card,"length_unit");
  if(modelUnits=="angstrom") {
    modelUnits="A";
  } else if(modelUnits=="bohr") {
    modelUnits="Bohr";
  }
  parse("LENGTH_UNITS",modelUnits);
  if(modelUnits.empty()) {
    error("the model has no card.json with a length_unit, please specify LENGTH_UNITS");
  }
  Units units;
  units.setLength(modelUnits);
  lengthScale=getUnits().getLength()/units.getLength();
  checkRead();

  for(const auto& p : model.parameters()) {
    dtype=p.scalar_type();
    break;
  }
  types=torch::tensor(std::vector<int64_t>(z.begin(),z.end()),torch::kInt64).to(device);

  log.printf("  LibTorch version: %d.%d.%d\n",TORCH_VERSION_MAJOR,TORCH_VERSION_MINOR,TORCH_VERSION_PATCH);
  log.printf("  model file: %s on device %s (%s)\n",fname.c_str(),devname.c_str(),dtype==torch::kFloat64?"float64":"float32");
  log.printf("  predicted property: %s [%s] (matches PROPERTY), cutoff %s %s\n",trainedProperty.c_str(),
             cardEntry(card,"units").c_str(),cardEntry(card,"cutoff").c_str(),modelUnits.c_str());
  // optional provenance of the trained model
  for(const auto& key : {"level_of_theory","weights_file","finetuned_from","training_data","notes"}) {
    const std::string value=cardEntry(card,key);
    if(!value.empty()) {
      log.printf("  %s: %s\n",key,value.c_str());
    }
  }
  log.printf("  export method: %s\n",cardEntry(card,"method").c_str());
  log.printf("  %zu atoms, positions multiplied by %g to convert to model units (%s)\n",atoms.size(),lengthScale,modelUnits.c_str());
  log.printf("  %s periodic boundary conditions to make the molecule whole\n",pbc?"using":"without");
  log.printf("  %s TYPES against the atomic masses at the first step\n",checkTypes?"checking":"not checking");

  // check that the model returns one value for this system (atoms on a line, 1 model unit apart)
  torch::Tensor testPos=torch::zeros({(int64_t)atoms.size(),3},torch::TensorOptions().dtype(dtype).device(device));
  testPos.select(1,0).copy_(torch::arange((int64_t)atoms.size(),testPos.options()));
  torch::jit::GraphOptimizerEnabledGuard noOptimization(false);
  auto test=model.forward({types,testPos}).toTensor();
  if(test.numel()!=1) {
    error("the model should return a single value, but returned " + std::to_string(test.numel()));
  }

  log<<"  Bibliography: "<<plumed.cite("Wang, Wang, Liu, Liu, Zhang, Shao, Liu, Nat. Commun. 15, 313 (2024)")<<"\n";

  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms(atoms);
}

// Stop if an atom's mass is not compatible with the atomic number given in TYPES
void VisnetModel::checkTypesAgainstMasses() {
  // standard atomic weights in amu
  static const std::map<int,double> standardMass{
    {1,1.008},{2,4.0026},{3,6.94},{4,9.0122},{5,10.81},{6,12.011},{7,14.007},{8,15.999},{9,18.998},
    {10,20.180},{11,22.990},{12,24.305},{13,26.982},{14,28.085},{15,30.974},{16,32.06},{17,35.45},
    {18,39.948},{19,39.098},{20,40.078},{35,79.904},{53,126.90}};
  const double tolerance=0.6;  // amu, less than the mass difference between neighboring elements
  const unsigned natoms=getNumberOfAtoms();
  // plumed driver sets the masses to NaN unless they are given with --mc or --pdb
  bool unavailable=true;
  for(unsigned i=0; i<natoms; i++) {
    unavailable = unavailable && (!std::isfinite(getMass(i)) || getMass(i)==1.0);
  }
  if(unavailable) {
    log.printf("  WARNING: the MD code did not provide the atomic masses, TYPES cannot be checked against them\n");
    return;
  }
  // masses lighter than hydrogen are not physical: the MD code passed invalid data, which is a
  // different problem from a wrong TYPES (e.g. ASE's Plumed calculator passes a temporary masses
  // array that may be freed before PLUMED reads it)
  std::string invalid;
  for(unsigned i=0; i<natoms; i++) {
    const double mass=getMass(i)*getUnits().getMass();
    if(!(mass>=0.5)) {
      invalid+="\n  atom "+std::to_string(getAbsoluteIndex(i).serial())+" has mass "+std::to_string(mass);
    }
  }
  if(!invalid.empty()) {
    error("the MD code passed non-physical masses, so TYPES cannot be checked against them:"+invalid+
          "\nThe masses passed to PLUMED must stay valid until the calculation is done (with ASE's Plumed calculator, keep"
          " a reference to the masses array). VISNET does not use the masses otherwise, so NOCHECK_TYPES can also be used.");
  }
  std::string mismatches;
  for(unsigned i=0; i<natoms; i++) {
    const double mass=getMass(i)*getUnits().getMass();
    auto expected=standardMass.find(atomicNumbers[i]);
    if(expected==standardMass.end()) {
      log.printf("  WARNING: no reference mass for atomic number %d, atom %d not checked\n",atomicNumbers[i],getAbsoluteIndex(i).serial());
    } else if(!(std::fabs(mass-expected->second)<=tolerance)) {
      mismatches+="\n  atom "+std::to_string(getAbsoluteIndex(i).serial())+" has mass "+std::to_string(mass)
                  +" but TYPES gives Z="+std::to_string(atomicNumbers[i])+" (mass "+std::to_string(expected->second)+")";
    }
  }
  if(!mismatches.empty()) {
    error("TYPES does not match the atomic masses, check that TYPES lists the atomic numbers in the same order as ATOMS:"
          +mismatches+"\nUse NOCHECK_TYPES if the masses were modified on purpose (e.g. hydrogen mass repartitioning).");
  }
  log.printf("  TYPES is consistent with the atomic masses\n");
}

void VisnetModel::calculate() {
  if(checkTypes) {
    checkTypesAgainstMasses();
    checkTypes=false;
  }
  if(pbc) {
    makeWhole();
  }
  const unsigned natoms=getNumberOfAtoms();
  const auto& positions=getPositions();
  // Vector is three contiguous doubles, so the positions can be viewed as a [natoms,3] tensor
  torch::Tensor pos=torch::from_blob(const_cast<Vector*>(positions.data()),{(int64_t)natoms,3},torch::kFloat64)
                    .to(device,dtype).set_requires_grad(true);

  // The optimizing graph executor specializes the traced graph on the first calls, which was found to give
  // wrong gradients (e.g. after a call with all atoms at the same position), so it is disabled here.
  torch::jit::GraphOptimizerEnabledGuard noOptimization(false);
  torch::Tensor cv=model.forward({types,pos*lengthScale}).toTensor();
  torch::Tensor grad;
  {
    ReleasePythonGIL noGIL;
    grad=torch::autograd::grad({cv.sum()},{pos})[0];
  }
  grad=grad.to(torch::kCPU,torch::kFloat64).contiguous();

  const auto g=grad.accessor<double,2>();
  for(unsigned i=0; i<natoms; i++) {
    setAtomsDerivatives(i,Vector(g[i][0],g[i][1],g[i][2]));
  }
  setBoxDerivativesNoPbc();
  setValue(cv.item<double>());
}

}
}

#endif
