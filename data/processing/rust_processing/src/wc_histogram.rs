use anyhow::Result;
use plotters::prelude::*;
use structopt::StructOpt;

mod utils;

#[derive(StructOpt, Debug)]
#[structopt(name = "processing")]
struct Opt {
    #[structopt(short, long)]
    input_file: String,

    #[structopt(short, long)]
    output_file: String,
}

fn main() -> Result<()> {
    let opt = Opt::from_args();
    let root = BitMapBackend::new(opt.output_file.as_str(), (640, 480)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut data = vec![];
    let mut max_elem = 0;
    if let Ok(lines) = utils::file_utils::read_lines(opt.input_file.as_str()) {
        for line in lines {
            let line = line?;

            let mut wc = 0;
            for _ in line.split_whitespace() {
                wc += 1;
            }
            max_elem = std::cmp::max(max_elem, wc);
            data.push(wc);
        }
    }
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(100)
        .y_label_area_size(100)
        .margin(5)
        .caption("Word count line distribution", ("sans-serif", 50))
        .build_cartesian_2d((0u32..300 as u32).into_segmented(), 0u32..max_elem as u32)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.1))
        .y_desc("Count")
        .x_desc("Bucket")
        .axis_desc_style(("sans-serif", 10))
        .draw()?;

    chart.draw_series(
        Histogram::vertical(&chart)
            .style(RED.mix(0.5).filled())
            .data(data.iter().map(|x: &u32| (*x, 1))),
    )?;
    Ok(())
}
