/// Some common tissue properties
///

pub mod tissuep5t {
    use crate::types::{Tissue, TissueProperties};

    pub fn get_tissue(tissue: Tissue) -> TissueProperties {
        match tissue {
            Tissue::WhiteMatter => TissueProperties {
                name: "wm",
                pd: 0.69,
                t1: 0.505,
                t2: 0.089,
                t2s: 0.07,
            },
            Tissue::GreyMatter => TissueProperties {
                name: "gm",
                pd: 0.82,
                t1: 0.775,
                t2: 0.110,
                t2s: 0.09,
            },
            Tissue::Caudate => TissueProperties {
                name: "caudate",
                pd: 0.82,
                t1: 0.505,
                t2: 0.089,
                t2s: 0.09,
            },
            Tissue::Thalamus => TissueProperties {
                name: "thalamus",
                pd: 0.82,
                t1: 0.730,
                t2: 0.1,
                t2s: 0.09,
            },
            Tissue::CerebroSpinalFluid => TissueProperties {
                name: "csf",
                pd: 1.0,
                t1: 4.0,
                t2: 2.0,
                t2s: 0.2,
            },
            Tissue::Blood => TissueProperties {
                name: "blood",
                pd: 1.0,
                t1: 1.12,
                t2: 0.26,
                t2s: 0.03,
            },
        }
    }
}
