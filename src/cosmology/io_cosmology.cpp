#ifdef COSMOLOGY

  #include <fstream>
  #include <iostream>

  #include "../cosmology/cosmology.h"
  #include "../io/io.h"

void Cosmology::Load_Scale_Outputs(struct Parameters *P)
{
  char filename_1[100];
  // create the filename to read from
  strcpy(filename_1, P->scale_outputs_file);
  chprintf(" Loading Scale_Factor Outputs: %s\n", filename_1);

  std::ifstream file_out(filename_1);
  std::string line;
  Real a_value;
  if (file_out.is_open()) {
    while (getline(file_out, line)) {
      a_value = atof(line.c_str());
      scale_outputs.push_back(a_value);
      n_outputs += 1;
      // chprintf("%f\n", a_value);
    }
    file_out.close();
    n_outputs        = scale_outputs.size();
    next_output_indx = 0;
    chprintf("  Loaded %d scale outputs \n", n_outputs);
  } else {
    chprintf("  Error: Unable to open cosmology outputs file\n");
    exit(1);
  }

  chprintf(" Setting next snapshot output\n");

  int scale_indx = next_output_indx;
  a_value        = scale_outputs[scale_indx];

  while ((current_a - a_value) > 1e-3) {
    // chprintf( "%f   %f\n", a_value, current_a);
    scale_indx += 1;
    a_value = scale_outputs[scale_indx];
  }
  next_output_indx = scale_indx;
  next_output      = a_value;
  chprintf("  Next output index: %d  \n", next_output_indx);
  chprintf("  Next output z value: %f  \n", 1. / next_output - 1);

  exit_now = false;
}

void Cosmology::Set_Scale_Outputs(struct Parameters *P)
{
  if (P->scale_outputs_file[0] == '\0') {
    chprintf(" Output every %d timesteps.\n", P->n_steps_output);
    Real scale_end = 1 / (P->End_redshift + 1);
    scale_outputs.push_back(current_a);
    scale_outputs.push_back(scale_end);
    n_outputs        = scale_outputs.size();
    next_output_indx = 0;
    next_output      = current_a;
    chprintf("  Next output index: %d  \n", next_output_indx);
    chprintf("  Next output z value: %f  \n", 1. / next_output - 1);
  } else {
    Load_Scale_Outputs(P);
  }
}

void Cosmology::Set_Next_Scale_Output()
{
  int scale_indx = next_output_indx;
  Real a_value   = scale_outputs[scale_indx];
  // chprintf("Setting next output index. Current index: %d    n_outputs: %d ",
  // scale_indx, n_outputs);

  // if  ( ( scale_indx == 0 ) && ( abs(a_value - current_a )<1e-5 ) )scale_indx
  // = 1;
  scale_indx += 1;

  if (scale_indx < n_outputs) {
    a_value          = scale_outputs[scale_indx];
    next_output_indx = scale_indx;
    next_output      = a_value;
  } else {
    exit_now = true;
  }
}

void Cosmology::Load_DynamicalDE_EquationOfState(struct Parameters *P)
{
  if (P->wDE_file[0] == '\0') {
    chprintf("wDE_file not found in parameter file \n");
    exit(1);
  }

  chprintf("Loading wDE info... \n");

  char wDE_filename[MAXLEN];
  strcpy(wDE_filename, P->wDE_file);

  std::fstream in(wDE_filename);
  std::string line;
  std::vector<std::vector<float>> v;
  int i = 0;
  if (in.is_open()) {
    while (std::getline(in, line)) {
      if (line.find("#") == 0) continue;

      float value;
      std::stringstream ss(line);
      v.push_back(std::vector<float>());

      while (ss >> value) {
        v[i].push_back(value);
      }
      i += 1;
    }
    in.close();
  } else {
    chprintf(" Error: Unable to open DE equation of state file: %s\n", wDE_filename);
    exit(1);
  }
  int n_lines = i;

  for (i = 0; i < n_lines; i++) {
    dynamicalDE_table_z.push_back(v[i][0]);
	dynamicalDE_table_w.push_back(v[i][1]);
  }

  for (i = 0; i < n_lines - 1; i++) {
    if (dynamicalDE_table_z[i] > dynamicalDE_table_z[i + 1]) {
      chprintf(
          " ERROR: equation of state must be ordered such that redshift is increasing "
          "as the rows increase in the file\n",
          wDE_filename);
      exit(2);
    }
  }

  n_wDE_samples = n_lines;
  chprintf(" Loaded DE equation of state file : \n");
  chprintf("  N redshift values: %d \n", n_wDE_samples);
  chprintf("  z_min = %f    z_max = %f \n", dynamicalDE_table_z[0], dynamicalDE_table_z[n_wDE_samples - 1]);
  chprintf("  w(z_min) = %f    w(z_max) = %f \n", dynamicalDE_table_w[0], dynamicalDE_table_w[n_wDE_samples - 1]);

  if (dynamicalDE_table_z[0] != 0.) {
    chprintf("We require z_min = 0 so that w(z=0) is well defined \n");
    exit(1);
  }
}
#endif
