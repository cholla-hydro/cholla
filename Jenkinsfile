pipeline
{
    agent none

    environment
    {
        CHOLLA_ROOT           = "${env.WORKSPACE}"
        CHOLLA_MACHINE        = 'crc'
        CHOLLA_LAUNCH_COMMAND = 'mpirun -np'
    }

    stages
    {
        stage('BuildAndTest')
        {
            matrix
            {
                agent
                {
                    label
                    {
                        label 'eschneider-ppc-n4'
                        customWorkspace "${env.JOB_NAME}/${env.CHOLLA_MAKE_TYPE}"
                    }
                }

                axes
                {
                    axis
                    {
                        name 'CHOLLA_MAKE_TYPE'
                        values 'hydro', 'gravity', 'disk', 'particles', 'cosmology', 'mhd', 'dust'
                    }
                }

                stages
                {
                    stage('Clone Repo Cholla')
                    {
                        steps
                        {
                            sh  '''
                                git submodule update --init --recursive
                                make clobber
                                '''
                        }
                    }
                    stage('Build Cholla')
                    {
                        steps
                        {
                            sh  '''
                                source builds/run_tests.sh
                                setupTests -c gcc -t ${CHOLLA_MAKE_TYPE}

                                buildCholla OPTIMIZE
                                '''
                        }
                    }
                    stage('Build Tests')
                    {
                        steps
                        {
                            sh  '''
                                source builds/run_tests.sh
                                setupTests -c gcc -t ${CHOLLA_MAKE_TYPE}

                                buildChollaTests
                                '''
                        }
                    }
                    stage('Run Tests')
                    {
                        steps
                        {
                            sh  '''
                                source builds/run_tests.sh
                                setupTests -c gcc -t ${CHOLLA_MAKE_TYPE}

                                runTests
                                '''
                        }
                    }
                    stage('Run Clang Tidy')
                    {
                        steps
                        {
                            catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                                sh  '''
                                    source builds/run_tests.sh
                                    setupTests -c gcc -t ${CHOLLA_MAKE_TYPE}

                                    module load clang/15.0.2
                                    make tidy CLANG_TIDY_ARGS="--warnings-as-errors=*" TYPE=${CHOLLA_MAKE_TYPE}
                                    '''
                            }
                        }
                    }
                    stage('Show Tidy Results')
                    {
                        steps
                        {
                            // Print the clang-tidy results with bars of equal
                            // signs seperating each file
                            sh  '''
                                printf '=%.0s' {1..100}
                                printf "\n"
                                cat tidy_results_cpp.log
                                printf '=%.0s' {1..100}
                                printf "\n"
                                cat tidy_results_gpu.log
                                printf '=%.0s' {1..100}
                                printf "\n"
                                '''
                        }
                    }
                }
            }
        }
    }
}
