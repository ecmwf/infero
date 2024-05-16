#
# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    """
    A simple model that takes a 10-dimensional input and outputs a single value
    """
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def generate_model(model_name):
    """
    Generates a simple model and saves it to a file
    """

    # Instantiate the model
    model = SimpleModel()

    example = torch.range(0, 9)
    traced_script_module = torch.jit.trace(model, example)

    # Save the torch model to a file
    # torch.save(model.state_dict(), "torch_model_python.pt")

    # Save the traced model to a file
    traced_script_module.save(model_name)



def run_model(model_name):
    """
    Loads the model and runs it
    """

    # Load the model
    model = torch.load(model_name)

    # test the model
    example = torch.range(0, 9)
    output = model.forward(example)
    print("output {:.8f} ".format(output.item() ))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script that generates/runs a simple torch model')
    parser.add_argument('--generate', action="store_true", default=False, help='Generate the simple model')
    parser.add_argument('--run', action="store_true", help='runs the generated \"torch_model.pt\"')
    args = parser.parse_args()

    traced_model_name = "torch_model.pt"

    if args.generate:
        print("Generating model {traced_model_name}...")
        generate_model(traced_model_name)

    if args.run:
        print("Running model {traced_model_name}...")
        run_model(traced_model_name)

    print("All done.")

