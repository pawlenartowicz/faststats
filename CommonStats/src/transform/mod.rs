//! Data transformations: ranking, normal scores, power transforms, PIT family.

pub mod rank;
pub mod normal_scores;
pub mod quantile_normalize;
pub mod power;
#[cfg(feature = "dist")]
pub mod pit;

pub use rank::{rank, Ties};
pub use normal_scores::normal_scores;
pub use quantile_normalize::quantile_normalize;
pub use power::{box_cox, yeo_johnson};
#[cfg(feature = "dist")]
pub use pit::{inv_pit, pit, quantile_map};
