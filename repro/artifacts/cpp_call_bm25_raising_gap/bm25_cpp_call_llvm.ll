; ModuleID = '/enzyme_call/source.cpp'
source_filename = "/enzyme_call/source.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem: none)
define dso_local void @entry(ptr noalias noundef readonly captures(none) %outs, ptr noalias noundef readonly captures(none) %ins) local_unnamed_addr #0 {
entry:
  %0 = load ptr, ptr %outs, align 8, !tbaa !5
  %1 = load ptr, ptr %ins, align 8, !tbaa !5
  %arrayidx2 = getelementptr inbounds nuw i8, ptr %ins, i64 8
  %2 = load ptr, ptr %arrayidx2, align 8, !tbaa !5
  %arrayidx3 = getelementptr inbounds nuw i8, ptr %ins, i64 16
  %3 = load ptr, ptr %arrayidx3, align 8, !tbaa !5
  %arrayidx4 = getelementptr inbounds nuw i8, ptr %ins, i64 24
  %4 = load ptr, ptr %arrayidx4, align 8, !tbaa !5
  br label %for.body.i

for.body.i:                                       ; preds = %for.cond.cleanup4.i, %entry
  %d.037.i = phi i64 [ 0, %entry ], [ %inc18.i, %for.cond.cleanup4.i ]
  %arrayidx.i.i = getelementptr inbounds nuw [4 x i8], ptr %3, i64 %d.037.i
  %5 = load float, ptr %arrayidx.i.i, align 4, !tbaa !7
  %mul.i = fmul float %5, 7.500000e-01
  %div.i = fdiv float %mul.i, 7.000000e+00
  %add.i = fadd float %div.i, 2.500000e-01
  %mul1.i = fmul float %add.i, 0x3FF3333340000000
  %arrayidx.i30.i = getelementptr inbounds nuw [20 x i8], ptr %1, i64 %d.037.i
  br label %for.body5.i

for.cond.cleanup4.i:                              ; preds = %for.body5.i
  %arrayidx.i34.i = getelementptr inbounds nuw [4 x i8], ptr %0, i64 %d.037.i
  store float %add15.i, ptr %arrayidx.i34.i, align 4, !tbaa !7
  %inc18.i = add nuw nsw i64 %d.037.i, 1
  %exitcond38.not.i = icmp eq i64 %inc18.i, 4
  br i1 %exitcond38.not.i, label %_Z10bm25_scoreILm4ELm5EEvRN6enzyme6tensorIfJXT_EEEERKNS1_IfJXT_EXT0_EEEERKNS1_IfJXT0_EEEERKS2_S9_.exit, label %for.body.i, !llvm.loop !9

for.body5.i:                                      ; preds = %for.body5.i, %for.body.i
  %score.036.i = phi float [ 0.000000e+00, %for.body.i ], [ %add15.i, %for.body5.i ]
  %t.035.i = phi i64 [ 0, %for.body.i ], [ %inc.i, %for.body5.i ]
  %arrayidx.i33.i = getelementptr inbounds nuw [4 x i8], ptr %arrayidx.i30.i, i64 %t.035.i
  %6 = load float, ptr %arrayidx.i33.i, align 4, !tbaa !7
  %add8.i = fadd float %mul1.i, %6
  %mul9.i = fmul float %6, 0x40019999A0000000
  %div10.i = fdiv float %mul9.i, %add8.i
  %arrayidx.i32.i = getelementptr inbounds nuw [4 x i8], ptr %2, i64 %t.035.i
  %7 = load float, ptr %arrayidx.i32.i, align 4, !tbaa !7
  %mul12.i = fmul float %7, %div10.i
  %arrayidx.i31.i = getelementptr inbounds nuw [4 x i8], ptr %4, i64 %t.035.i
  %8 = load float, ptr %arrayidx.i31.i, align 4, !tbaa !7
  %mul14.i = fmul float %8, %mul12.i
  %add15.i = fadd float %score.036.i, %mul14.i
  %inc.i = add nuw nsw i64 %t.035.i, 1
  %exitcond.not.i = icmp eq i64 %inc.i, 5
  br i1 %exitcond.not.i, label %for.cond.cleanup4.i, label %for.body5.i, !llvm.loop !12

_Z10bm25_scoreILm4ELm5EEvRN6enzyme6tensorIfJXT_EEEERKNS1_IfJXT_EXT0_EEEERKNS1_IfJXT0_EEEERKS2_S9_.exit: ; preds = %for.cond.cleanup4.i
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem: none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.ident = !{!0}
!llvm.errno.tbaa = !{!1}

!0 = !{!"clang version 23.0.0git"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !3, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"float", !3, i64 0}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
!12 = distinct !{!12, !10, !11}
