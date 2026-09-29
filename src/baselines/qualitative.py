from __future__ import annotations

from pathlib import Path

import numpy as np

from baselines.scholler import predict as scholler_predict


def constant_velocity(xy:np.ndarray,history:int=8,future:int=12)->np.ndarray:
    if history != 8 or future != 12:
        observed = np.asarray(xy[:history, 0])
        delta = observed[-1] - observed[-2]
        return observed[-1] + np.arange(1, future + 1)[:, None] * delta
    return scholler_predict(np.asarray(xy[:history, 0]))


def trajectory_metrics(pred:np.ndarray,gt:np.ndarray)->tuple[float,float]:
    distance=np.linalg.norm(pred-gt,axis=-1);return float(distance.mean()),float(distance[-1])


def render_scene(
    xy:np.ndarray,
    valid:np.ndarray,
    predictions:np.ndarray,
    output:Path,
    title:str="",
    *,
    history:int=8,
    future:int=12,
) -> dict:
    """Render one coherent best-ADE model sample alongside CV and ground truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xy=np.asarray(xy);valid=np.asarray(valid);predictions=np.asarray(predictions)
    if predictions.ndim==2:predictions=predictions[None]
    gt=xy[history:history+future,0];cv=constant_velocity(xy,history,future)
    model_ades=np.linalg.norm(predictions-gt[None],axis=-1).mean(axis=1);chosen=int(model_ades.argmin());model=predictions[chosen]
    model_ade,model_fde=trajectory_metrics(model,gt);cv_ade,cv_fde=trajectory_metrics(cv,gt)
    points=[xy[valid],cv,model];bounds=np.concatenate(points);padding=1.0
    fig,axis=plt.subplots(figsize=(7.2,6.2));axis.set_aspect("equal",adjustable="box")
    for agent in range(1,xy.shape[1]):
        mask=valid[:,agent];path=xy[:,agent][mask]
        if len(path):axis.plot(path[:,0],path[:,1],color="0.65",linewidth=1.0,alpha=.75)
    observed=xy[:history,0];axis.plot(observed[:,0],observed[:,1],color="tab:blue",linewidth=2.4)
    axis.scatter(gt[:,0],gt[:,1],color="tab:orange",s=22,zorder=4)
    start=observed[-1:]
    axis.plot(*np.vstack([start,cv]).T,color="tab:green",linestyle="--",linewidth=1.8)
    axis.plot(*np.vstack([start,model]).T,color="tab:red",linewidth=2.0)
    axis.set_xlim(bounds[:,0].min()-padding,bounds[:,0].max()+padding);axis.set_ylim(bounds[:,1].min()-padding,bounds[:,1].max()+padding)
    axis.set_xlabel("x (m)");axis.set_ylabel("y (m)");axis.grid(alpha=.2)
    if title:axis.set_title(title)
    axis.legend(handles=[
        Line2D([0],[0],color="tab:blue",label="Past observation"),
        Line2D([0],[0],color="0.65",label="Context pedestrian trajectories"),
        Line2D([0],[0],marker="o",linestyle="None",color="tab:orange",label="Future ground truth"),
        Line2D([0],[0],color="tab:green",linestyle="--",label=f"Constant Velocity (ADE {cv_ade:.2f} m, FDE {cv_fde:.2f} m)"),
        Line2D([0],[0],color="tab:red",label=f"Model best-of-{len(predictions)} (ADE {model_ade:.2f} m, FDE {model_fde:.2f} m)"),
    ],loc="upper center",bbox_to_anchor=(.5,-.12),frameon=False,fontsize=8)
    fig.tight_layout();output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix(".png"),dpi=180,bbox_inches="tight");fig.savefig(output.with_suffix(".pdf"),bbox_inches="tight");plt.close(fig)
    return {"model_ade":model_ade,"model_fde":model_fde,"cv_ade":cv_ade,"cv_fde":cv_fde,"selected_sample":chosen,"k":len(predictions)}


def contact_sheet(images:list[Path],output:Path)->None:
    if not images:return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns=min(4,len(images));rows=int(np.ceil(len(images)/columns));fig,axes=plt.subplots(rows,columns,figsize=(4*columns,3.5*rows),squeeze=False)
    for axis,path in zip(axes.flat,images):axis.imshow(plt.imread(path));axis.axis("off")
    for axis in axes.flat[len(images):]:axis.axis("off")
    fig.tight_layout();output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=150,bbox_inches="tight");plt.close(fig)
