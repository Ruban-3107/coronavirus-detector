
pipeline {
    agent any
    
    environment {
        NEW_VERSION ='1.3.0'
    }

    stages {
        stage('build') {
            steps {
                echo 'build stage.....'
            }
        }
       stage('test') {
           when {
               expression {
                   BRANCH_NAME =='master'
                   
                   
               }
           }
            steps {
                echo 'test stage.....'
                echo "running in version ${env.NEW_VERSION}"
            }
        }
       stage('deploy') {
            steps {
                echo 'deploy stage.....'
            }
        }
    }
}
