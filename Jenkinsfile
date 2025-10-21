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
                        //values 'hydro', 'gravity', 'disk', 'particles', 'cosmology', 'mhd', 'dust', 'cooling'
                        values 'gravity', 'cosmology', 'mhd'
                    }
                }

                stages
                {
                    stage('Clone Repo Cholla')
                    {
                        steps
                        {
                            sh  '''
                                git config --list --show-origin
                                if [ "${CHOLLA_MAKE_TYPE}" = "cosmology" ] ||
                                   [ "${CHOLLA_MAKE_TYPE}" = "mhd" ] ||
                                   [ "${CHOLLA_MAKE_TYPE}" = "gravity" ]; then
                                    #./tools/ci-setup-submodule.py \
                                    #   --color \
                                    #   --simulate-lfs-fetch-failure \
                                    #   --fallback-manual-lfs-download
                                    GIT_LFS_SKIP_SMUDGE=1 git submodule update --init
                                    git status
                                    git -C ./cholla-tests-data status
                                    GIT_LFS_SKIP_SMUDGE=1 git -C ./cholla-tests-data restore .
                                    git -C ./cholla-tests-data status
                                    cat ./cholla-tests-data/initial_conditions/tCOSMOLOGYSYSTEM50Mpc_CorrectInputExpectCorrectOutput/0.h5.0
                                    git submodule foreach --recursive git lfs fetch
                                    git submodule foreach --recursive git lfs checkout
                                    git -C ./cholla-tests-data status

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
                            retry(2)
                            {
                                sh  '''
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
                                sh  '''
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
                            sh  '''
                                printf '=%.0s' {1..100}
                                printf "\n"
                                cat tidy_results_cpp_${CHOLLA_MAKE_TYPE}.log
                                printf '=%.0s' {1..100}
                                printf "\n"
                                cat tidy_results_gpu_${CHOLLA_MAKE_TYPE}.log
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
