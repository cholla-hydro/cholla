#if defined(PARTICLES) && defined(PARTICLES_GPU)

  #ifndef PARTICLES_BOUNDARIES_H
    #define PARTICLES_BOUNDARIES_H

part_int_t Select_Particles_to_Transfer_GPU_function(part_int_t n_local, int side, Real domainMin, Real domainMax,
                                                     Real *pos_d, int *n_transfer_d, int *n_transfer_h,
                                                     bool *transfer_flags_d, int *transfer_indices_d,
                                                     int *replace_indices_d, int *transfer_prefix_sum_d,
                                                     int *transfer_prefix_sum_blocks_d);

void Load_Particles_to_Transfer_GPU_function(int n_transfer, int field_id, int n_fields_to_transfer, Real *field_d,
                                             int *transfer_indices_d, Real *send_buffer_d, Real domainMin,
                                             Real domainMax, int boundary_type);
void Load_Particles_to_Transfer_Int_GPU_function(int n_transfer, int field_id, int n_fields_to_transfer,
                                                 part_int_t *field_d, int *transfer_indices_d, Real *send_buffer_d,
                                                 Real domainMin, Real domainMax, int boundary_type);

void Replace_Transfered_Particles_GPU_function(int n_transfer, Real *field_d, int *transfer_indices_d,
                                               int *replace_indices_d, bool print_replace);
void Replace_Transfered_Particles_Int_GPU_function(int n_transfer, part_int_t *field_d, int *transfer_indices_d,
                                                   int *replace_indices_d, bool print_replace);

void Copy_Particles_GPU_Buffer_to_Host_Buffer(int n_transfer, Real *buffer_h, Real *buffer_d);

void Copy_Particles_Host_Buffer_to_GPU_Buffer(int n_transfer, Real *buffer_h, Real *buffer_d);

void Unload_Particles_to_Transfer_GPU_function(int n_local, int n_transfer, int field_id, int n_fields_to_transfer,
                                               Real *field_d, Real *recv_buffer_d);
void Unload_Particles_Int_to_Transfer_GPU_function(int n_local, int n_transfer, int field_id, int n_fields_to_transfer,
                                                   part_int_t *field_d, Real *recv_buffer_d);

  #endif  // PARTICLES_H
#endif    // PARTICLES