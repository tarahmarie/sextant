#!/usr/bin/env bash

set -euo pipefail

#Let's make a list of projects
PROJECTS=()
while IFS='' read -r line; do PROJECTS+=("$line"); done < <(basename -a ./projects/*/ | paste -d '\n' -s -)

#Some helper variables
project_file_count=""
last_run_file_count=""
# Define colors and styles using ANSI escape codes
GREEN='\033[1;32m'
RED='\033[1;31m'
BOLD='\033[1m'
RESET='\033[0m'

set_up_database() {
    python -c "from database_ops import create_db_and_tables; create_db_and_tables()"
    python -c "from predict_ops import setup_auto_author_prediction_tables; setup_auto_author_prediction_tables()"
}

main_menu() {
    tput clear;
    printf "\n\tsextant\n\n"
    printf "Choose an option\n\n"

    COLUMNS=20

    select _ in "Work on an existing project" "Create a new project/collection of texts" "Quit"
    do
        case "${REPLY}" in
            1)
                choose_project
                return
                ;;
            2)
                initialize_new_project
                return
                ;;
            3)
                exit 0
                ;;
        esac
    done
}

#Kicks off at the start. Sets up a new project, or moves you on to picking an existing project.
#A project targets batches of texts, of any kind. This block asks to be pointed at the /splits dir 
#to find texts in the format it needs, and looks for the relevant inputs like alignments.

initialize_new_project () {
    tput clear;
    printf "\n\tHello!\n\n\t"

    read -rp "Would you like to prepare the folders for a new project: (y/n) " new_project_choice
    
    local lower_choice
    lower_choice=$(echo "$new_project_choice" | awk '{print tolower($0)}')

    if [ "$lower_choice" == "y" ]; then
        printf "\t"
        read -rp "What should I name the project? " new_project_name
        printf "\t"
        read -rp "Is $new_project_name correct? (y/n) " confirm_new_project_name
        
        local confirm_new_project_name_choice
        confirm_new_project_name_choice=$(echo "$confirm_new_project_name" | awk '{print tolower($0)}')

        if [ "$confirm_new_project_name_choice" == "y" ]; then
            printf "\n\tOK, making project..."
            mkdir -p ./projects/"$new_project_name"/{db,alignments,splits,results,visualizations}
            printf "\n\n\tDirectories created."
            set_up_database
            printf "\n\n\tDatabase set up."
            printf "\n\n\tAdd your alignments to ./projects/%s/alignments/" "$new_project_name"
            printf "\n\tAdd your split files to ./projects/%s/splits/" "$new_project_name"
            printf "\n\n"
            exit 0
        elif [ "$confirm_new_project_name_choice" == "n" ]; then
            printf "\n\tOK, quitting..."
            exit 0
        else
            initialize_new_project
        fi        

    elif [ "$lower_choice" == "n" ]; then
        exit 0
    else
        initialize_new_project
    fi
}

find_alignment_file () {
    local project=$1

    tput clear

    local alignment_files
    alignment_files=$(find "projects/${project}/alignments/" -type f -name "*.jsonl" -exec basename {} \; )
    filecount=$(echo "${alignment_files}" | wc -l)

    case $filecount in
        0)
            printf "\n\tNo alignment files found. Please add them to ./projects/%s/alignments/" "$project"
            exit 1
            ;;
        1)
            echo "${alignment_files}" > .alignments_file_name
            return
            ;;
        *)
            COLUMNS=20
            select selection in ${alignment_files}
            do
                if [ "${selection}" == "quit" ]; then
                    exit 0
                elif [ "${selection}" == "go back" ]; then
                    choose_project
                    return
                else
                    echo "${selection}" > .alignments_file_name
                    return 
                fi
            done
            ;;
    esac
}

#Presents you a list of existing projects and gets your selection for what to work with.
choose_project () {
    tput clear;
    printf "\n\nHere are the existing projects you can work on. Select one:\n\n"

    COLUMNS=20
    select dir in "${PROJECTS[@]}"
    do
        case "${dir}" in
            quit)
                exit 0
                ;;
            'go back')
                main_menu
                return
                ;;
            *)
                echo "${dir}" > .current_project
                find_alignment_file "${dir}"
                check_file_counts
                return
                ;;
        esac
    done

    check_file_counts
}

#Compares file counts with last run.  If match, asks if you want to re-do.
#If there's not a match (e.g. if you've included an extra piece of text or there's
#an alignment file missing, etc), it does not ask you anything, but continues
#to run the work script with no further prompt. Intended to save you from having to 
#rerun the statistics each time if no changes to the texts have occurred, and 
#will display the previous stats generated.

check_file_counts () {
    project_name=$(cat .current_project)
    project_file_count=$(find ./projects/"$project_name"/splits -type f ! -name '.DS_Store' | wc -l | awk '{print $1}')
    
    if [ -f "projects/$project_name/db/$project_name.db" ]; then
        last_run_file_count=$(sqlite3 -readonly -batch -line ./projects/"$project_name"/db/"$project_name".db "SELECT number_files FROM last_run;" | awk '{print $3}')
    fi

    #Let's see if we need to do anything:
    printf "\n\tNumber of files in project: %s" "$project_file_count"
    printf "\n\tNumber of files in last run: %s" "$last_run_file_count"

    if [ "$project_file_count" == "$last_run_file_count" ]; then
        printf "\n\n"
        read -rp "File count matches from last run. Did you want to run everything again? (y/n) " run_again_choice
        local lower_choice
        lower_choice=$(echo "$run_again_choice" | awk '{print tolower($0)}')
        if [ "$lower_choice" == "y" ]; then
            load_from_db
        elif [ "$lower_choice" == "n" ]; then
            tput clear;
            printf "\n\nOK, here are the stats from the previous run..."
            python show_previous_averages.py;
        else
            check_file_counts
        fi
    else
        load_from_db
    fi
}

#Actually executes the work on a given project. From previous function, doing this
#only if either asked to or if the file count doesn't match from a previous run.
#Deletes old databases, runs the set of functions on files for calculation of
#results and statistics, and stores in fresh dbs.

load_from_db () {
    printf "\n\n\tRemoving old dbs (if they exist) and loading data..."
    printf "\n"

    if [ -f "./projects/$project_name/db/$project_name.db" ]; then
        rm "./projects/$project_name/db/$project_name.db";
    fi

    if [ -f "./projects/$project_name/db/$project_name-predictions.db" ]; then
        rm "./projects/$project_name/db/$project_name-predictions.db";
    fi

    # Ensure db folder exists
    if [ ! -d "./projects/$project_name/db" ]; then
        mkdir -p "./projects/$project_name/db";
    fi    

    python3 init_db.py;
    python3 load_authors_and_texts.py; # go find all the relevant texts & pair them up.
    python3 load_alignments.py "$(cat .alignments_file_name)"; 
    python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
    python3 load_hapaxes.py;
    python3 load_hapax_intersects.py;
    python3 load_relationships.py;
    python3 load_jaccard.py;
    python3 do_svm.py;
    #python do_svm_burrows_delta.py;
    #python3 auto_author_prediction.py; #vestigial from stepwise threshhold grid search exploration, replaced by logistic regression
    python3 logistic_regression.py;
}

#Check if we're starting a new project.
main_menu