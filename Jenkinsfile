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
                        values 'hydro', 'gravity', 'disk', 'particles', 'cosmology', 'mhd', 'dust', 'cooling'
                    }
                }

                stages
                {
                    stage('Clone Repo Cholla')
                    {
                        steps
                        {
                            sh  '''#!/bin/bash -e
                                # enable tracing mode now that the shell
                                # configuration has been read
                                set -x

                                if [ "${CHOLLA_MAKE_TYPE}" = "cosmology" ] ||
                                   [ "${CHOLLA_MAKE_TYPE}" = "mhd" ] ||
                                   [ "${CHOLLA_MAKE_TYPE}" = "hydro" ] ||
                                   [ "${CHOLLA_MAKE_TYPE}" = "gravity" ]; then
                                    ./tools/ci-setup-submodule.py \
                                       --color \
                                       --fallback-manual-lfs-download
                                else
                                    # we skip the download because it's not currently
                                    # necessary & we want to minimize calls to
                                    # downloads from GitHub's raw-urls (when git-lfs
                                    # commonly fails)
                                    echo "hard-coded to skip submodule download"
                                fi
                                make clobber
                                '''
                        }
                    }
                    stage('Build Cholla')
                    {
                        steps
                        {
                            sh  '''#!/bin/bash -e
                                # enable tracing mode now that the shell
                                # configuration has been read
                                set -x

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
                            sh  '''#!/bin/bash -e
                                # enable tracing mode now that the shell
                                # configuration has been read
                                set -x

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
                            retry(2)
                            {
                                sh  '''#!/bin/bash -e
                                    # enable tracing mode now that the shell
                                    # configuration has been read
                                    set -x

                                    source builds/run_tests.sh
                                    setupTests -c gcc -t ${CHOLLA_MAKE_TYPE}

                                    runTests
                                    '''
                            }
                        }
                    }
                    stage('Run Clang Tidy')
                    {
                        steps
                        {
                            catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                                sh  '''#!/bin/bash -e
                                    # enable tracing mode now that the shell
                                    # configuration has been read
                                    set -x

                                    source builds/run_tests.sh
                                    setupTests -c gcc -t ${CHOLLA_MAKE_TYPE}

                                    module load clang/17.0.1
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
                            sh  '''#!/bin/bash -e
                                # we explicitly choose not to use tracing mode

                                echo "tidy_results_cpp_${CHOLLA_MAKE_TYPE}.log"
                                printf '=%.0s' {1..100}
                                printf "\n"
                                cat tidy_results_cpp_${CHOLLA_MAKE_TYPE}.log
                                printf "\n\n"

                                echo "tidy_results_gpu_${CHOLLA_MAKE_TYPE}.log"
                                printf '=%.0s' {1..100}
                                printf "\n"
                                cat tidy_results_gpu_${CHOLLA_MAKE_TYPE}.log
                                '''
                        }
                    }
                }
            }
        }
    }
}
