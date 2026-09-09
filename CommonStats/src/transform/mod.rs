//! Data transformations: ranking, normal scores, power transforms, PIT family.

pub mod normal_scores;
#[cfg(feature = "dist")]
pub mod pit;
pub mod power;
pub mod quantile_normalize;
pub mod rank;

pub use normal_scores::normal_scores;
#[cfg(feature = "dist")]
pub use pit::{inv_pit, pit, quantile_map};
pub use power::{box_cox, yeo_johnson};
pub use quantile_normalize::quantile_normalize;
pub use rank::{Ties, rank};
