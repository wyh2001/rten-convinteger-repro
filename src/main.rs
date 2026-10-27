use std::error::Error;

use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use rten::ValueView;

const MODEL: &str = "repro.onnx";

fn main() -> Result<(), Box<dyn Error>> {
    let input_data = vec![11u8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
    let input = Array4::from_shape_vec((1, 3, 2, 2), input_data.clone())?;

    let mut session = Session::builder()?.commit_from_file(MODEL)?;
    let tensor = Tensor::from_array(input)?;
    let outputs = session.run(ort::inputs![tensor])?;
    let (_, ort_output) = outputs[0].try_extract_tensor::<i32>()?;

    let model = rten::Model::load_file(MODEL)?;
    let input_id = model.node_id("x")?;
    let output_id = model.node_id("y")?;
    let input_value = ValueView::from_shape([1, 3, 2, 2], &input_data)?;
    let [rten_output] = model.run_n(vec![(input_id, input_value.into())], [output_id], None)?;
    let (_, rten_output) = rten_output.into_shape_vec::<i32, 4>()?;

    println!("expected: [78, 78, 78, 78]");
    println!("ort:      {:?}", ort_output);
    println!("rten:     {:?}", rten_output);

    Ok(())
}
