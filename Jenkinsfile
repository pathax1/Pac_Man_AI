pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                git credentialsId: 'a295e9de-c839-4351-bbaa-00b0640e3151',
                    branch: 'main',
                    url: 'git@github.com:pathax1/Pac_Man_AI.git'
            }
        }

        stage('Setup Pythons Environment') {
            steps {
                sh 'python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('Train RL Models') {
            steps {
                sh 'python3 main_train.py'
            }
        }

        stage('Save Trained Models') {
            steps {
                archiveArtifacts artifacts: '*.pth, *.pkl', fingerprint: true
            }
        }

        stage('Notify Completion') {
            steps {
                mail to: 'your-email@example.com',
                    subject: 'Pac-Man RL Training Completed',
                    body: 'All RL models have finished training on Jenkins!'
            }
        }
    }
}
