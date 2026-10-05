mod decoder;
mod models;
mod ops;
mod runtime;
mod vision;

fn main() {
    models::dispatch(std::env::args().skip(1).collect());
}
