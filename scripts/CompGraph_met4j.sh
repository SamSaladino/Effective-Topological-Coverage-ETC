#!/bin/bash

echo "This is the good file"

# Set path to data
path_to_data=$(pwd)
resources_dir="$path_to_data/resources"
generated_dir="$resources_dir/generated"

# Define input files
sbml_file="$resources_dir/Human-GEM.xml"
side_compounds_file="$resources_dir/sidecomp.txt"



# Define output filenames
output_graph1="$generated_dir/compound_graph_no_sidecompounds.gml"
output_graph2="$generated_dir/compound_graph_merged_compartments.gml"

# Define the Java command
java_cmd_sidecomp="java -cp met4j-toolbox-2.0.1-SNAPSHOT.jar fr.inrae.toulouse.metexplore.met4j_toolbox.networkAnalysis.SideCompoundsScan"
java_cmd="java -cp met4j-toolbox-2.0.1-SNAPSHOT.jar fr.inrae.toulouse.metexplore.met4j_toolbox.convert.Sbml2CompoundGraph"

# Function to generate a graph
generate_graph() {
    local description=$1
    local additional_args=$2
    local output_file=$3

    echo "Generating $description"
    $java_cmd -i "$sbml_file" -sc "$side_compounds_file" -ri -un -o "$output_file" -f gml -me $additional_args
}

# Go to the met4j root folder
echo "Going to the root folder of met4j"
cd ~/met4j/met4j-toolbox/target

echo "The repository with the human data is at $resources_dir"

# Generate list of side compounds
echo "Sacaning for side compounds..."
$java_cmd_sidecomp -i $sbml_file -cc -uf -m by_name -s -o $side_compounds_file
echo "Side compounds have been rewritten in $side_compounds_file"

# Generate graphs
generate_graph "first compound graph (removing side compounds, undirected, removing isolated nodes)" "" "$output_graph1"
generate_graph "second compound graph (removing side compounds, merging compartments, undirected, removing isolated nodes)" "-mc by_name" "$output_graph2"

echo "Graphs have been generated!"
echo "Check them in the 'generated' folder at $generated_dir"