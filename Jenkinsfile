pipeline {
    agent any

    environment {
        PYTHON_VENV = "venv"
        TRAINED_MODEL = "trained_pacman.h5"
        GIT_SSH_COMMAND = "ssh -i C:/Users/Autom/.ssh/id_rsa -o StrictHostKeyChecking=no"
    }

    stages {

        stage('Checkout Code') {
            steps {
                script {
                    env.GIT_SSH_COMMAND = "ssh -i C:/Users/Autom/.ssh/id_rsa -o StrictHostKeyChecking=no"
                }
                git branch: 'master',
                    credentialsId: 'PacManSSH',
                    url: 'git@github.com:pathax1/Pac_Man_AI.git'
            }
        }

        stage('Set up Python Environment') {
            steps {
                bat """
                python -m venv ${PYTHON_VENV}
                call ${PYTHON_VENV}\\Scripts\\activate.bat
                pip install --upgrade pip
                if exist requirements.txt (
                    pip install -r requirements.txt
                ) else (
                    echo No requirements.txt found, skipping dependency installation.
                )
                """
            }
        }

        stage('Train Pac-Man Model') {
            steps {
                bat """
                call ${PYTHON_VENV}\\Scripts\\activate.bat
                python train.py --episodes 1000 --save_path ${TRAINED_MODEL}
                """
            }
        }

        stage('Archive Trained Model') {
            steps {
                archiveArtifacts artifacts: "${TRAINED_MODEL}", fingerprint: true
            }
        }
    }

    post {
        always {
            echo "Build and training process complete."
        }
        success {
            echo "Training succeeded!"
        }
        failure {
            echo "Training failed. Check logs for details."
        }
    }
}
