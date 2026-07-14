template <enzyme::size_t Rows, enzyme::size_t Cols>
void transpose(enzyme::tensor<float, Cols, Rows> &out,
               const enzyme::tensor<float, Rows, Cols> &input) {
  for (enzyme::size_t row = 0; row < Rows; ++row)
    for (enzyme::size_t col = 0; col < Cols; ++col)
      out[col][row] = input[row][col];
}
