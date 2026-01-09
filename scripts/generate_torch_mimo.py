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


# Define a custom module
class MultiInputNet(nn.Module):

    def __init__(self):

        super(MultiInputNet, self).__init__()

        # Define submodules for the two inputs
        self.input1_fc = nn.Linear(10, 20)
        self.input2_fc = nn.Linear(5, 20)

        # Define a module to combine the features
        self.combined_fc = nn.Linear(40, 1)
    
    def forward(self, inputs_dict):

        # Process each input through its respective layer
        x1 = torch.relu(self.input1_fc(inputs_dict["input_1"]))
        x2 = torch.relu(self.input2_fc(inputs_dict["input_2"]))

        # Concatenate the outputs
        combined = torch.cat((x1, x2), dim=1)

        # Final layer
        output_1 = torch.sigmoid(self.combined_fc(combined))
        output_2 = torch.tanh(self.combined_fc(combined))

        # pack the outputs into a dictionary
        outputs = {"output_1": output_1, "output_2": output_2}

        return outputs


def generate_model(model_name):

    # Create an instance of the model
    model = MultiInputNet()

    # Example inputs
    input1 = torch.range(0, 9).reshape(1, 10)
    input2 = torch.range(0, 4).reshape(1, 5)

    # ------- generate the model script -------
    inputs = {'input_1': input1, 'input_2': input2}
    traced_script_module = torch.jit.trace(model, inputs, strict=False)

    # Save the torch model to a file
    # torch.save(model.state_dict(), "torch_model_python.pt")

    # Save the traced model to a file
    traced_script_module.save(model_name)
    # -----------------------------------------

    return model


def run_model(model_name):
    """
    Loads the model and runs it
    """

    # Load the model
    model = torch.load(model_name)

    # test the model
    input1 = torch.range(0, 9).reshape(1, 10)
    input2 = torch.range(0, 4).reshape(1, 5)
    example_inputs = {"input_1": input1, "input_2": input2}

    outputs_dict = model.forward(example_inputs)

    print(f"model: \n{model}")
    print(f"model.graph: \n{model.graph}")
    print(f"output #1 from python model {outputs_dict["output_1"]}")
    print(f"output #2 from python model {outputs_dict["output_2"]}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script that generates/runs a simple torch model')
    parser.add_argument('--generate', action="store_true", default=False, help='Generate the simple model')
    parser.add_argument('--run', action="store_true", help='runs the generated \"torch_mimo.pt\"')
    args = parser.parse_args()

    traced_model_name = "torch_mimo.pt"

    if args.generate:
        print("Generating model {traced_model_name}...")
        generate_model(traced_model_name)

    if args.run:
        print("Running model {traced_model_name}...")
        run_model(traced_model_name)

    print("All done.")

