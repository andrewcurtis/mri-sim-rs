pub mod fse;
pub mod se;
pub mod fid;
pub mod space;

pub enum SequenceSelection {
    FSE(fse::FseParams),
    SE(se::SeParams),
    FID(fid::FidParams),
    SPACE(space::SpaceParams),
    }


