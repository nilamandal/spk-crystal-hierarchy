from spektral.data import DisjointLoader
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import networkx as nx

from aqsol_experiments import Dataset_from_json
from spektral_essential_objects import AtomFeaDataset, NotShrinking


def mean_squared_error(real, pred):
    return np.mean((real - pred) ** 2)


def visualize_molecule_pooling(
    atoms,
    bonds,
    assign,
    molecule_name,
    savepath
):
    """
    Creates:
    1. Molecular graph colored by hard pool assignment
    2. Scatterplot of soft assignment space
    """

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    ###################################################
    # GRAPH VISUALIZATION
    ###################################################

    G = nx.Graph()

    positions = {}
    colors = []
    labels = {}

    for k, atom in enumerate(atoms):

        G.add_node(k)

        positions[k] = (
            atom["x"],
            atom["y"]
        )

        labels[k] = atom["element"]

        hard_cluster = np.argmax(assign[k])
        colors.append(hard_cluster)

    for bond in bonds:

        u = bond["aid1"] - 1
        v = bond["aid2"] - 1

        G.add_edge(u, v)

    plt.figure(figsize=(8, 8))

    nx.draw(
        G,
        pos=positions,
        labels=labels,
        with_labels=True,
        node_color=colors,
        cmap=plt.cm.Set1,
        node_size=700,
        font_size=10
    )

    plt.title(f"{molecule_name} Pool Assignments")

    plt.savefig(
        os.path.join(
            savepath,
            molecule_name + "_graph_pool.png"
        )
    )

    plt.close()

    ###################################################
    # ASSIGNMENT SPACE SCATTER
    ###################################################

    plt.figure(figsize=(6, 6))

    assignment_np = np.array(assign)

    elements = [atom["element"] for atom in atoms]
    unique_elements = list(set(elements))

    color_map = {}

    cmap = plt.cm.tab10

    for idx, el in enumerate(unique_elements):
        color_map[el] = cmap(idx)

    for k, atom in enumerate(atoms):

        element = atom["element"]

        plt.scatter(
            assignment_np[k, 0],
            assignment_np[k, 1],
            color=color_map[element],
            s=120,
            label=element if k == elements.index(element) else ""
        )

        plt.text(
            assignment_np[k, 0],
            assignment_np[k, 1],
            str(k),
            fontsize=9
        )

    plt.xlabel("Pool 1 Assignment")
    plt.ylabel("Pool 2 Assignment")

    plt.title(f"{molecule_name} Assignment Space")

    plt.legend()

    plt.savefig(
        os.path.join(
            savepath,
            molecule_name + "_assignment_space.png"
        )
    )

    plt.close()


def save_pool_csv(
    atoms,
    assign,
    molecule_name,
    savepath
):

    rows = []

    for k, atom in enumerate(atoms):

        row = {
            "atom_id": k,
            "element": atom["element"],
            "x": atom["x"],
            "y": atom["y"],
            "pool1": float(assign[k][0]),
            "pool2": float(assign[k][1]),
            "hard_assignment": int(np.argmax(assign[k]))
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(
        savepath,
        molecule_name + "_pooling.csv"
    )

    df.to_csv(csv_path, index=False)


def evaluate(
    loader,
    model,
    df,
    write_output_path
):

    output = []

    step = 0
    molecule_index = 0

    while step < loader.steps_per_epoch:

        step += 1

        inputs, target = loader.__next__()

        pred, s_tensor = model(inputs, training=False)

        batch_size = len(s_tensor)

        for j in range(batch_size):

            row = df.iloc[molecule_index]

            molecule_name = f"molecule_{molecule_index}"

            try:
                structure_json = json.loads(
                    row["structure"].replace("'", '"')
                )
            except Exception as e:
                print("JSON parse failed")
                print(e)

                molecule_index += 1
                continue

            atoms = structure_json["atoms"]
            bonds = structure_json["bonds"]

            assign = s_tensor[j]

            assign = assign[:len(atoms)]

            prediction = pred[j]
            
            target_value = target[j]

            abs_error = np.abs(prediction - target_value)

            print("\n========================")
            print("MOLECULE:", molecule_name)
            print("Prediction:", prediction)
            print("Target:", target_value)
            print("Absolute Error:", abs_error)
            print("========================\n")

            molecule_save_path = os.path.join(
                write_output_path,
                molecule_name
            )

            if not os.path.exists(molecule_save_path):
                os.makedirs(molecule_save_path)

            ###################################################
            # SAVE CSV
            ###################################################

            save_pool_csv(
                atoms,
                assign,
                molecule_name,
                molecule_save_path
            )

            ###################################################
            # VISUALIZE POOLING
            ###################################################

            visualize_molecule_pooling(
                atoms,
                bonds,
                assign,
                molecule_name,
                molecule_save_path
            )

            ###################################################
            # SAVE PREDICTION INFO
            ###################################################

            metrics_df = pd.DataFrame({
                "prediction": [prediction],
                "target": [target_value],
                "absolute_error": [abs_error]
            })

            metrics_df.to_csv(
                os.path.join(
                    molecule_save_path,
                    "metrics.csv"
                ),
                index=False
            )

            molecule_index += 1

        mse = tf.reduce_mean((target - pred) ** 2)

        output.append(float(mse))

    output = np.array(output)

    return {
        "MSE": np.mean(output),
        "RMSE": np.sqrt(np.mean(output))
    }


def main(
    fullpath_of_model,
    fullpath_of_data_file,
    write_output_path,
    parampath
):

    ###################################################
    # LOAD CONFIG
    ###################################################

    config = json.load(
        open(
            os.path.join(
                parampath,
                "params.json"
            )
        )
    )

    ###################################################
    # LOAD DATA
    ###################################################

    val_df = pd.read_csv(
        fullpath_of_data_file,
        header=0
    )

    data_dir = os.path.dirname(fullpath_of_data_file)

    dataset = Dataset_from_json(
        val_df,
        'r'
    )

    loader = DisjointLoader(
        dataset,
        shuffle=False,
        batch_size=16
    )

    ###################################################
    # BUILD MODEL
    ###################################################

    model = NotShrinking(
        'r',
        1,
        config['embedding_size'],
        config['cgcnn_num'],
        config['cgcnn_num2'],
        softmax_beta=config['softmax_beta'],
        k=2
    )

    ###################################################
    # ENABLE RETURN OF ASSIGNMENT MATRIX
    ###################################################

    model.return_s = True

    ###################################################
    # BUILD MODEL ON DUMMY INPUT
    ###################################################

    temp_loader = DisjointLoader(
        dataset,
        batch_size=2
    )

    temp_inputs, temp_targets = temp_loader.__next__()

    pred, s_tensor = model(
        temp_inputs,
        training=False
    )

    ###################################################
    # LOAD WEIGHTS
    ###################################################
    weights_path = os.path.join(
        fullpath_of_model,
        "4goodmodel.ckpt.index"
    )
    checkpoint_dir = os.path.dirname(weights_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    ###################################################
    # OUTPUT DIR
    ###################################################

    if not os.path.exists(write_output_path):
        os.makedirs(write_output_path)

    ###################################################
    # EVALUATE + VISUALIZE
    ###################################################

    results = evaluate(
        loader,
        model,
        val_df,
        write_output_path
    )

    print("\nFINAL RESULTS")
    print(results)

    return results


if __name__ == "__main__":

    ###################################################
    # EXAMPLE PATHS
    ###################################################

    fullpath_of_model = "../ray_results/main_workflow_2026-04-29_18-28-07/main_workflow_bc7c8006_1_trial_index=0,batch_size=64,cgcnn_num=2,cgcnn_num2=2,embedding_size=4,lr=0.0166,softmax_beta=119574.7390_2026-04-29_18-28-10"
    fullpath_of_data_file = "./aqsol_test.csv"

    parampath_for_model = fullpath_of_model

    write_output_path = os.path.join(
        fullpath_of_model,
        "pool_visualizations"
    )

    main(
        fullpath_of_model,
        fullpath_of_data_file,
        write_output_path,
        parampath_for_model
    )
