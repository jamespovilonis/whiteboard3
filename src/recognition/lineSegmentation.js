const DEFAULT_CONFIG = Object.freeze({
  idleDelayMs: 1000,
  horizontalPadding: 50,
  verticalPadding: 10,
  strictMinVerticalOverlapRatio: 0.15,
  detectedBandMinVerticalOverlapRatio: 0.25,
  rowCenterThresholdRatio: 0.82,
  maxSearchStates: 8192
});

export function segmentMathLines(strokes, options = {}) {
  const config = { ...DEFAULT_CONFIG, ...(options.config || {}) };
  const ignoredStrokeIds = new Set((options.ignoredStrokeIds || []).map(String));
  for (const stroke of [
    ...detectEnclosingAnnotationStrokes(strokes || []),
    ...detectSeparatedTopAnnotationStrokes(strokes || []),
    ...detectTinyIsolatedScratchStrokes(strokes || [])
  ]) {
    ignoredStrokeIds.add(String(stroke.id));
  }
  const eligibleStrokes = filterStrokes(strokes || [], options.answerBox)
    .filter((stroke) => !ignoredStrokeIds.has(String(stroke.id)));
  const candidates = [];
  const byStrokeSet = new Map();

  const addCandidate = (candidateStrokes, profile, extra = {}) => {
    const usable = uniqueStrokes(candidateStrokes).filter((stroke) => stroke?.canvasBbox);
    if (usable.length === 0) return null;

    const strokeSetKey = strokeSetKeyFor(usable);
    let candidate = byStrokeSet.get(strokeSetKey);
    if (!candidate) {
      candidate = candidateFromStrokes(usable, {
        profile,
        profiles: [profile],
        config,
        ...extra
      });
      byStrokeSet.set(strokeSetKey, candidate);
      candidates.push(candidate);
    } else {
      if (!candidate.profiles.includes(profile)) candidate.profiles.push(profile);
      candidate.profile = candidate.profiles[0];
      candidate.sources = uniqueStrings([...(candidate.sources || []), ...(extra.sources || [profile])]);
      if (extra.parentCandidateId) candidate.parentCandidateId = extra.parentCandidateId;
    }
    return candidate;
  };

  if (eligibleStrokes.length === 0) {
    return emptySegmentation();
  }

  const parent = addCandidate(eligibleStrokes, 'parent', { sources: ['all-strokes'] });
  const baseGroups = buildBaseGroups(eligibleStrokes, config);
  for (const stack of buildTallFractionStackGroups(clusterStrokeRows(parent, config), config)) {
    addCandidate(stack.strokes, 'fraction-stack-line', {
      sources: ['global-tall-fraction-stack'],
      parentCandidateId: parent?.candidateId || null
    });
  }

  for (const group of buildOverlapGroups(eligibleStrokes, 'loose', config)) {
    addCandidate(group.strokes, 'loose', { sources: ['loose-overlap'] });
  }
  for (const group of buildOverlapGroups(eligibleStrokes, 'strict', config)) {
    addCandidate(group.strokes, 'strict', { sources: ['strict-overlap'] });
  }
  for (const group of baseGroups) {
    addCandidate(group.strokes, 'temporal', { sources: ['temporal-catchment'] });
  }

  for (const group of baseGroups) {
    const groupCandidate = candidateFromStrokes(group.strokes, { profile: 'row-parent', config });
    const rawRows = clusterStrokeRows(groupCandidate, config);
    if (rawRows.length > 1) {
      for (const row of rawRows) {
        addCandidate(row.strokes, 'raw-row-line', {
          sources: ['stroke-row-raw'],
          parentCandidateId: groupCandidate.candidateId
        });
        for (const splitRow of buildCenterGapLineGroups(row, config)) {
          addCandidate(splitRow.strokes, 'row-line', {
            sources: ['center-gap-row'],
            parentCandidateId: groupCandidate.candidateId
          });
        }
      }
      for (const stack of buildCompactFractionStackGroups(rawRows, config)) {
        addCandidate(stack.strokes, 'fraction-stack-line', {
          sources: ['fraction-stack'],
          parentCandidateId: groupCandidate.candidateId
        });
      }
      for (const stack of buildTallFractionStackGroups(rawRows, config)) {
        addCandidate(stack.strokes, 'fraction-stack-line', {
          sources: ['tall-fraction-stack'],
          parentCandidateId: groupCandidate.candidateId
        });
      }
      for (const superscriptLine of buildSuperscriptLineGroups(rawRows, config)) {
        addCandidate(superscriptLine.strokes, 'superscript-line', {
          sources: ['superscript-line'],
          parentCandidateId: groupCandidate.candidateId
        });
      }
    }
    const rows = splitCandidateIntoRows(groupCandidate, config);
    if (rows.length > 1) {
      for (const row of rows) {
        addCandidate(row.strokes, 'row-line', {
          sources: ['stroke-row'],
          parentCandidateId: groupCandidate.candidateId
        });
        for (const splitRow of buildCenterGapLineGroups(row, config)) {
          addCandidate(splitRow.strokes, 'row-line', {
            sources: ['center-gap-row'],
            parentCandidateId: groupCandidate.candidateId
          });
        }
      }
    } else {
      addCandidate(group.strokes, 'row-line', { sources: ['stroke-row-single'] });
      for (const splitRow of buildCenterGapLineGroups(groupCandidate, config)) {
        addCandidate(splitRow.strokes, 'row-line', {
          sources: ['center-gap-row'],
          parentCandidateId: groupCandidate.candidateId
        });
      }
    }
  }

  for (const group of buildProjectionLineGroups(eligibleStrokes)) {
    addCandidate(group.strokes, 'projection-line', { sources: ['vertical-projection'] });
  }

  const detectedBands = clusterDetections(
    options.detections || [],
    config.detectedBandMinVerticalOverlapRatio
  );
  if (detectedBands.length > 0) {
    for (const group of baseGroups) {
      const dbnetParent = addCandidate(group.strokes, 'dbnet-parent', { sources: ['dbnet-parent'] });
      const split = splitCandidateByDetections(dbnetParent, detectedBands, config);
      if (split.length > 1) {
        for (const line of split) {
          addCandidate(line.strokes, 'dbnet-line', {
            sources: ['dbnet-line'],
            parentCandidateId: dbnetParent.candidateId
          });
        }
      }
    }
  }

  for (const stroke of eligibleStrokes) {
    addCandidate([stroke], 'fallback-stroke', { sources: ['fallback'] });
  }

  assignConflicts(candidates);
  const selected = selectCandidateCover(candidates, {
    maxSearchStates: config.maxSearchStates
  });

  return {
    candidates: sortCandidates(candidates),
    selected,
    partitions: partitionCandidateIds(candidates),
    parentCandidateId: parent?.candidateId || null
  };
}

export function selectCandidateCover(candidates, options = {}) {
  const evidenceScores = normalizeEvidenceScores(options.scoreByCandidateId);
  const entries = (candidates || [])
    .filter((candidate) => candidate?.strokeIds?.length)
    .map((candidate) => ({
      candidate,
      selectionScore: scoreCandidateGeometry(candidate, candidates) +
        (evidenceScores.get(candidate.candidateId) || 0)
    }));

  if (entries.length === 0) return [];

  const allStrokeIds = uniqueStrings(entries.flatMap((entry) => entry.candidate.strokeIds)).sort();
  const byStroke = new Map();
  for (const entry of entries) {
    for (const strokeId of entry.candidate.strokeIds) {
      if (!byStroke.has(strokeId)) byStroke.set(strokeId, []);
      byStroke.get(strokeId).push(entry);
    }
  }

  let best = null;
  let bestScore = -Infinity;
  let explored = 0;
  const memo = new Map();
  const maxSearchStates = Number(options.maxSearchStates || DEFAULT_CONFIG.maxSearchStates);

  function search(covered, coveredCount, selected, score) {
    if (explored++ > maxSearchStates) return;
    if (coveredCount === allStrokeIds.length) {
      if (
        score > bestScore ||
        (score === bestScore && (!best || selected.length < best.length))
      ) {
        bestScore = score;
        best = selected.slice();
      }
      return;
    }

    const memoKey = allStrokeIds.filter((id) => covered.has(id)).join('|');
    if (memo.has(memoKey) && memo.get(memoKey) >= score) return;
    memo.set(memoKey, score);

    let choices = null;
    for (const strokeId of allStrokeIds) {
      if (covered.has(strokeId)) continue;
      const available = (byStroke.get(strokeId) || []).filter((entry) => (
        !selected.some((chosen) => candidatesOverlap(entry.candidate, chosen.candidate))
      ));
      if (choices === null || available.length < choices.length) {
        choices = available;
      }
    }

    if (!choices || choices.length === 0) return;
    choices.sort((a, b) => b.selectionScore - a.selectionScore);

    for (const choice of choices) {
      const nextCovered = new Set(covered);
      let added = 0;
      for (const strokeId of choice.candidate.strokeIds) {
        if (!nextCovered.has(strokeId)) {
          nextCovered.add(strokeId);
          added += 1;
        }
      }
      selected.push(choice);
      search(nextCovered, coveredCount + added, selected, score + choice.selectionScore);
      selected.pop();
    }
  }

  search(new Set(), 0, [], 0);

  const selected = sortCandidates((best || fallbackCover(entries)).map((entry) => entry.candidate));
  if (options.baselineCandidates?.length) {
    return stabilizeEvidenceSelection(selected, options.baselineCandidates, {
      allCandidates: candidates,
      evidenceScores,
      maxLineCountIncrease: options.maxEvidenceLineCountIncrease,
      minSplitImprovement: options.minEvidenceSplitImprovement,
      minMergeImprovement: options.minEvidenceMergeImprovement,
      maxSplitPieces: options.maxEvidenceSplitPieces
    });
  }
  return selected;
}

export function scoreCandidateGeometry(candidate, allCandidates = []) {
  const profiles = candidate.profiles || [candidate.profile || 'unknown'];
  let score = -0.22;

  if (profiles.includes('dbnet-line')) score += 3.1;
  if (profiles.includes('row-line')) score += 2.6;
  if (profiles.includes('fraction-stack-line')) score += 4.8;
  if (profiles.includes('raw-row-line')) score += 1.95;
  if (profiles.includes('projection-line')) score += 0.45;
  if (profiles.includes('strict')) score += 1.2;
  if (profiles.includes('loose')) score += 0.5;
  if (profiles.includes('temporal')) score += 0.35;
  if (profiles.includes('parent')) score -= 1.4;
  if (profiles.includes('row-parent')) score -= 1.0;
  if (profiles.includes('dbnet-parent')) score -= 1.2;
  if (profiles.includes('fallback-stroke')) score -= 5.0;

  const height = bboxHeight(candidate.tightBbox);
  const width = bboxWidth(candidate.tightBbox);
  const medianHeight = median(candidate.strokes.map(strokeHeight)) || 1;
  const spread = height / medianHeight;
  const rawRows = clusterStrokeRows(candidate, DEFAULT_CONFIG);
  const structuralRows = splitCandidateIntoRows(candidate, DEFAULT_CONFIG);
  const nearbyCompactFraction = bboxHeight(candidate.tightBbox) <= 140 &&
    hasNearbyCompactFractionBridge(candidate, rawRows);
  const plusMinusStructure = hasAttachedPlusMinusStroke(candidate, rawRows);
  const builtTallFractionStack = profiles.includes('fraction-stack-line') &&
    bboxHeight(candidate.tightBbox) > 150 &&
    bboxHeight(candidate.tightBbox) <= 285 &&
    candidate.strokes.length <= 12 &&
    rawRows.length >= 2 &&
    rawRows.length <= 6 &&
    hasWideFractionBar(candidate);
  const fractionBridge = hasFractionLikeBridge(candidate) ||
    hasInlineFractionBridge(candidate) ||
    nearbyCompactFraction ||
    hasCompactStackedFractionColumn(candidate, rawRows) ||
    containsFractionBridge(candidate, rawRows) ||
    containsFractionBridge(candidate, structuralRows);
  const compactFractionLine = !parentLikeCandidateProfiles(profiles) &&
    bboxHeight(candidate.tightBbox) <= 140 &&
    hasCompactFractionStructure(candidate);
  const superscriptStructural = profiles.includes('superscript-line') &&
    hasSuperscriptStructure(rawRows, medianHeight);
  const fractionStructural = fractionBridge || compactFractionLine || builtTallFractionStack;
  const structural = fractionBridge ||
    compactFractionLine ||
    superscriptStructural ||
    plusMinusStructure ||
    structuralRows.some((row) => rowHasTallOperatorStroke(row, rowMedianHeight(row)));
  const parentLike = profiles.includes('parent') || profiles.includes('dbnet-parent') || profiles.includes('row-parent');

  if (candidate.strokes.length <= 1) score -= 1.8;
  if (candidate.strokes.length <= 2 && width < medianHeight * 4) score -= 0.8;
  if (structural && candidate.strokes.length > 2) score += 0.85;
  if (
    rawRows.length > 1 &&
    !structural &&
    isLineLikeCandidate(candidate) &&
    hasLargeInternalRowGap(rawRows, medianHeight)
  ) {
    score -= Math.min(7.5, (rawRows.length - 1) * 3.4);
  }
  const localFractionBridge = fractionBridge && rawRows.length > 1 && rawRows.length <= 3 && spread <= 5.2;
  if (localFractionBridge && !parentLike) {
    score += Math.min(8.5, (rawRows.length - 1) * 4.1);
  }
  if (compactFractionLine) score += 4.2;
  if (profiles.includes('fraction-stack-line') && fractionStructural) score += 3.2;
  if (superscriptStructural) score += 9.7;
  if (builtTallFractionStack) score += 27.5;
  if (plusMinusStructure) score += 6.8;
  score += inlineFragmentContinuityBonus(candidate, allCandidates, { medianHeight });
  if (profiles.includes('raw-row-line') && hasCompactFractionContainer(candidate, allCandidates)) {
    score -= 8;
  }

  if (spread > 2.7) {
    score -= (spread - 2.7) * (structural ? 0.16 : 0.65);
  }

  if (parentLike) {
    const children = allCandidates.filter((child) => (
      child !== candidate &&
      child.strokeIds?.length &&
      strokeSetContains(candidate, child) &&
      (child.profiles || []).some((profile) => profile.endsWith('-line') || profile === 'strict')
    ));
    if (childLinesCoverParent(candidate, children) && children.length > 1) {
      score -= (children.length - 1) * (structural ? 0.75 : 4.5);
    }
  }

  if (profiles.length === 1 && profiles.includes('projection-line')) {
    const children = allCandidates.filter((child) => (
      child !== candidate &&
      child.strokeIds?.length &&
      (child.strokeIds || []).length < (candidate.strokeIds || []).length &&
      strokeSetContains(candidate, child) &&
      isNonProjectionLineCandidate(child)
    ));
    if (childLinesCoverParent(candidate, children) && children.length > 1) {
      score -= (children.length - 1) * 12;
    }
  }

  if (
    (profiles.includes('loose') || profiles.includes('projection-line')) &&
    !profiles.includes('strict') &&
    !profiles.includes('row-line') &&
    !profiles.includes('raw-row-line')
  ) {
    const children = allCandidates.filter((child) => (
      child !== candidate &&
      child.strokeIds?.length &&
      (child.strokeIds || []).length < (candidate.strokeIds || []).length &&
      strokeSetContains(candidate, child) &&
      isPreferredChildLineCandidate(child)
    ));
    if (childLinesCoverParent(candidate, children) && children.length > 1) {
      const independentRows = childrenLookLikeIndependentRows(children, width, medianHeight, candidate);
      const protectedFractionStack = profiles.includes('fraction-stack-line') && fractionStructural;
      score -= (children.length - 1) * (
        protectedFractionStack ? 0.8 : (structural && !independentRows ? 1.0 : 8.5)
      );
    }
  }

  if (
    !parentLike &&
    (profiles.includes('row-line') || profiles.includes('raw-row-line') || profiles.includes('dbnet-line'))
  ) {
    const children = allCandidates.filter((child) => (
      child !== candidate &&
      child.strokeIds?.length &&
      (child.strokeIds || []).length < (candidate.strokeIds || []).length &&
      strokeSetContains(candidate, child) &&
      isIndependentLineChildCandidate(child)
    ));
    if (childLinesCoverParent(candidate, children) && children.length > 1) {
      const independentRows = childrenLookLikeIndependentRows(children, width, medianHeight, candidate);
      const protectedFractionStack = profiles.includes('fraction-stack-line') && fractionStructural;
      if ((independentRows && !protectedFractionStack) || !structural) {
        score -= (children.length - 1) * (independentRows ? 9.8 : 4.2);
      }
    }
  }
  if (
    profiles.length === 1 &&
    profiles.includes('dbnet-line') &&
    !fractionStructural
  ) {
    const children = independentChildRowsNearDbnetBand(candidate, allCandidates, { medianHeight });
    if (childrenLookLikeIndependentRows(children, width, medianHeight, candidate)) {
      score -= Math.min(16, rawRows.length * 5.2);
    }
  }

  if (profiles.includes('loose') && !profiles.includes('strict')) {
    const strictComponents = allCandidates.filter((child) => (
      child !== candidate &&
      child.profiles?.includes('strict') &&
      strokeSetContains(candidate, child)
    )).length;
    if (strictComponents > 1) score -= (strictComponents - 1) * 0.8;
  }

  score -= adjacentLineIntrusionPenalty(candidate, allCandidates, { medianHeight });
  score -= dbnetLineBoundaryIntrusionPenalty(candidate, allCandidates);
  score -= dbnetBoundaryStrokeIntrusionPenalty(candidate, allCandidates, { medianHeight });
  score -= superscriptSplitChildPenalty(candidate, allCandidates);
  score -= annotationPenalty(candidate);

  return score;
}

export function scoreRecognitionEvidence(candidate, predictions = {}, context = {}) {
  const result = predictions.comer || predictions;
  if (!result || result.failed || result.timedOut) return -1000;
  const candidates = result.candidates || [];
  const top = result.top || candidates[0] || result;
  const latex = String(top.latex || '').trim();
  if (!latex) return -50;

  const tokenCount = Math.max(1, latex.split(/\s+/).filter(Boolean).length);
  const modelScore = typeof top.score === 'number' ? Math.max(-20, Math.min(2, top.score / tokenCount)) : -4;
  let score = modelScore * 0.35;

  score += bracesAreBalanced(latex) ? 0.25 : -4;
  if (/[=<>]/.test(latex)) score += 0.2;
  if (/[+\-=^_]\s*$/.test(latex) || /^\s*[=+]/.test(latex)) score -= 2;
  if (/\\(frac|sum|int|prod|sqrt|begin)|[\^_][\s{]/.test(latex)) score += 1.3;

  const elapsed = Number(result.elapsedSeconds);
  if (Number.isFinite(elapsed) && elapsed > 3.5) {
    score -= Math.min(2.5, (elapsed - 3.5) * 0.22);
    score -= slowMultiRowPenalty(candidate, elapsed);
  }

  const referenceSymbols = extractMathSymbols([
    context.problemLatex,
    ...(context.previousLatex || [])
  ].filter(Boolean).join(' '));
  if (referenceSymbols.size > 0) {
    const candidateSymbols = extractMathSymbols(latex);
    const shared = [...candidateSymbols].filter((symbol) => referenceSymbols.has(symbol)).length;
    score += Math.min(0.8, shared * 0.12);
  }

  return score;
}

function slowMultiRowPenalty(candidate, elapsedSeconds) {
  if (!candidate?.strokes?.length) return 0;
  const rows = clusterStrokeRows(candidate, DEFAULT_CONFIG);
  if (rows.length <= 1) return 0;
  if (isStructuralMathCandidate(candidate)) return 0;

  const profiles = candidate.profiles || [];
  const parentLike = profiles.includes('parent') ||
    profiles.includes('dbnet-parent') ||
    profiles.includes('row-parent') ||
    profiles.includes('loose') ||
    profiles.includes('temporal');
  if (!parentLike && rows.length <= 2) return 0;

  const medianHeight = median(candidate.strokes.map(strokeHeight)) || 1;
  const spread = bboxHeight(candidate.tightBbox) / medianHeight;
  const elapsedOver = Math.max(0, Number(elapsedSeconds) - 3.5);
  const rowPenalty = Math.max(0, rows.length - 1) * 0.85;
  const spreadPenalty = Math.max(0, spread - 3.2) * 0.35;
  return Math.min(6.5, rowPenalty + spreadPenalty + elapsedOver * 0.55);
}

function stabilizeEvidenceSelection(proposedCover, baselineCover, options = {}) {
  const proposed = sortCandidates(proposedCover || []);
  const baseline = sortCandidates(baselineCover || []);
  if (baseline.length === 0 || proposed.length === 0) return proposed;
  if (coverSignature(proposed) === coverSignature(baseline)) return proposed;

  const maxLineCountIncrease = Number.isFinite(Number(options.maxLineCountIncrease))
    ? Number(options.maxLineCountIncrease)
    : 1;
  if (proposed.length > baseline.length + maxLineCountIncrease) {
    return baseline;
  }

  if (fragmentsBaselineCover(proposed, baseline, options)) {
    return baseline;
  }

  if (coalescesBaselineCover(proposed, baseline, options)) {
    return baseline;
  }

  return proposed;
}

function fragmentsBaselineCover(proposed, baseline, options = {}) {
  const evidenceScores = options.evidenceScores || new Map();
  const allCandidates = options.allCandidates || [];
  const maxSplitPieces = Number.isFinite(Number(options.maxSplitPieces))
    ? Number(options.maxSplitPieces)
    : 2;
  const minSplitImprovement = Number.isFinite(Number(options.minSplitImprovement))
    ? Number(options.minSplitImprovement)
    : 8;

  for (const base of baseline) {
    const pieces = proposed.filter((candidate) => (
      candidate.candidateId !== base.candidateId &&
      strokeSetContains(base, candidate)
    ));
    if (pieces.length <= 1) continue;
    if (!childLinesCoverParent(base, pieces)) continue;
    if (pieces.length > maxSplitPieces) return true;

    const baseScore = evidenceAdjustedCandidateScore(base, evidenceScores, allCandidates);
    const pieceScore = pieces.reduce((sum, piece) => (
      sum + evidenceAdjustedCandidateScore(piece, evidenceScores, allCandidates)
    ), 0);
    if (baselineUnreadWithStrongRecognizedPiece(base, pieces, evidenceScores)) {
      continue;
    }
    if (isStructuralMathCandidate(base)) return true;
    if (!pieces.every(isEvidenceLineCandidate)) return true;

    const requiredGain = minSplitImprovement + (pieces.length - 1) * 2;
    if (pieceScore - baseScore < requiredGain) return true;
  }
  return false;
}

function baselineUnreadWithStrongRecognizedPiece(base, pieces, evidenceScores) {
  const baseEvidence = Number(evidenceScores.get(base.candidateId) || 0);
  if (baseEvidence > -20) return false;
  return (pieces || []).some((piece) => {
    const evidence = Number(evidenceScores.get(piece.candidateId) || 0);
    return evidence >= 4 && isEvidenceLineCandidate(piece);
  });
}

function coalescesBaselineCover(proposed, baseline, options = {}) {
  const evidenceScores = options.evidenceScores || new Map();
  const allCandidates = options.allCandidates || [];
  const minMergeImprovement = Number.isFinite(Number(options.minMergeImprovement))
    ? Number(options.minMergeImprovement)
    : 8;

  for (const candidate of proposed) {
    const pieces = baseline.filter((base) => (
      base.candidateId !== candidate.candidateId &&
      strokeSetContains(candidate, base)
    ));
    if (pieces.length <= 1) continue;
    if (!childLinesCoverParent(candidate, pieces)) continue;
    if (!evidenceScores.has(candidate.candidateId)) return true;

    const candidateEvidence = Number(evidenceScores.get(candidate.candidateId) || 0);
    if (candidateEvidence <= -20) return true;
    const pieceEvidence = pieces.reduce((sum, piece) => (
      sum + Number(evidenceScores.get(piece.candidateId) || 0)
    ), 0);
    if (candidateEvidence - pieceEvidence >= minMergeImprovement) continue;

    const candidateScore = evidenceAdjustedCandidateScore(candidate, evidenceScores, allCandidates);
    const pieceScore = pieces.reduce((sum, piece) => (
      sum + evidenceAdjustedCandidateScore(piece, evidenceScores, allCandidates)
    ), 0);
    const requiredGain = minMergeImprovement + (pieces.length - 1) * 2;
    if (candidateScore - pieceScore < requiredGain) return true;
  }
  return false;
}

function evidenceAdjustedCandidateScore(candidate, evidenceScores, allCandidates) {
  const evidence = Number(evidenceScores.get(candidate.candidateId) || 0);
  return scoreCandidateGeometry(candidate, allCandidates) + evidence;
}

function coverSignature(cover) {
  return (cover || [])
    .map((candidate) => uniqueStrings(candidate.strokeIds || []).sort().join('|'))
    .sort()
    .join('||');
}

function isEvidenceLineCandidate(candidate) {
  const profiles = candidate?.profiles || [];
  if (profiles.includes('fallback-stroke')) return false;
  return profiles.some((profile) => (
    profile === 'dbnet-line' ||
    profile === 'fraction-stack-line' ||
    profile === 'superscript-line' ||
    profile === 'row-line' ||
    profile === 'raw-row-line' ||
    profile === 'projection-line' ||
    profile === 'strict' ||
    profile === 'loose'
  ));
}

function isLineLikeCandidate(candidate) {
  const profiles = candidate?.profiles || [];
  if (profiles.includes('fallback-stroke')) return false;
  return profiles.some((profile) => (
    profile === 'dbnet-line' ||
    profile === 'fraction-stack-line' ||
    profile === 'superscript-line' ||
    profile === 'row-line' ||
    profile === 'raw-row-line' ||
    profile === 'projection-line' ||
    profile === 'strict' ||
    profile === 'loose'
  ));
}

function parentLikeCandidateProfiles(profiles = []) {
  return profiles.includes('parent') ||
    profiles.includes('dbnet-parent') ||
    profiles.includes('row-parent') ||
    profiles.includes('temporal');
}

function hasLargeInternalRowGap(rows, medianHeight) {
  const sortedRows = (rows || []).slice().sort(compareRows);
  if (sortedRows.length <= 1) return false;
  const threshold = Math.max(18, (medianHeight || 1) * 0.9);
  for (let index = 0; index < sortedRows.length - 1; index += 1) {
    if (verticalGap(sortedRows[index].bbox, sortedRows[index + 1].bbox) > threshold) {
      return true;
    }
  }
  return false;
}

function isNonProjectionLineCandidate(candidate) {
  const profiles = candidate?.profiles || [];
  if (profiles.includes('fallback-stroke')) return false;
  return profiles.some((profile) => (
    profile === 'dbnet-line' ||
    profile === 'fraction-stack-line' ||
    profile === 'superscript-line' ||
    profile === 'row-line' ||
    profile === 'raw-row-line' ||
    profile === 'strict' ||
    profile === 'loose'
  ));
}

function isIndependentLineChildCandidate(candidate) {
  const profiles = candidate?.profiles || [];
  if (profiles.includes('fallback-stroke')) return false;
  return profiles.some((profile) => (
    profile === 'dbnet-line' ||
    profile === 'fraction-stack-line' ||
    profile === 'superscript-line' ||
    profile === 'row-line' ||
    profile === 'raw-row-line' ||
    profile === 'projection-line' ||
    profile === 'strict'
  ));
}

function isPreferredChildLineCandidate(candidate) {
  const profiles = candidate?.profiles || [];
  if (profiles.includes('fallback-stroke')) return false;
  return profiles.some((profile) => (
    profile === 'dbnet-line' ||
    profile === 'fraction-stack-line' ||
    profile === 'superscript-line' ||
    profile === 'row-line' ||
    profile === 'raw-row-line' ||
    profile === 'strict'
  ));
}

function independentChildRowsNearDbnetBand(candidate, allCandidates, { medianHeight }) {
  const parentBox = candidate?.tightBbox || candidate?.bbox;
  if (!parentBox) return [];
  const parentWidth = bboxWidth(parentBox);
  const parentHeight = bboxHeight(parentBox);
  const verticalAllowance = Math.max(12, (medianHeight || 1) * 0.45);
  return (allCandidates || []).filter((child) => {
    if (child === candidate || !child?.strokeIds?.length) return false;
    const profiles = child.profiles || [];
    if (profiles.includes('fallback-stroke') || profiles.includes('dbnet-line')) return false;
    if (!isIndependentLineChildCandidate(child)) return false;
    const childBox = child.tightBbox || child.bbox;
    if (!childBox || bboxHeight(childBox) >= parentHeight * 0.82) return false;
    if (bboxWidth(childBox) < Math.max(80, parentWidth * 0.24)) return false;
    if (horizontalOverlapRatio(childBox, parentBox) < 0.38) return false;
    const childCenter = centerY(childBox);
    return childCenter >= parentBox.yMin - verticalAllowance &&
      childCenter <= parentBox.yMax + verticalAllowance;
  });
}

function childrenLookLikeIndependentRows(children, parentWidth, parentMedianHeight, parentCandidate = null) {
  const broadChildren = (children || [])
    .filter((child) => bboxWidth(child.tightBbox || child.bbox) >= Math.max(120, parentWidth * 0.45))
    .sort((a, b) => centerY(a.tightBbox || a.bbox) - centerY(b.tightBbox || b.bbox));
  if (broadChildren.length < 2) return false;
  const top = broadChildren[0];
  const bottom = broadChildren[broadChildren.length - 1];
  const centerSpan = centerY(bottom.tightBbox || bottom.bbox) - centerY(top.tightBbox || top.bbox);
  if (parentCandidate && hasFractionLikeBridge(parentCandidate)) {
    return centerSpan >= Math.max(72, (parentMedianHeight || 1) * 1.25);
  }
  return centerSpan >= Math.max(48, (parentMedianHeight || 1) * 1.15);
}

function adjacentLineIntrusionPenalty(candidate, allCandidates, { medianHeight }) {
  if (!candidate?.tightBbox) return 0;
  const profiles = candidate.profiles || [];
  if (!profiles.some((profile) => (
    profile === 'row-line' ||
    profile === 'raw-row-line' ||
    profile === 'strict' ||
    profile === 'loose' ||
    profile === 'projection-line'
  ))) {
    return 0;
  }

  const threshold = Math.max(8, (medianHeight || 1) * 0.12);
  let penalty = 0;
  for (const neighbor of allCandidates || []) {
    if (neighbor === candidate || !neighbor?.tightBbox) continue;
    if (candidatesOverlap(candidate, neighbor)) continue;
    if (!isPreferredChildLineCandidate(neighbor)) continue;
    if (centerY(neighbor.tightBbox) <= centerY(candidate.tightBbox)) continue;
    if (horizontalOverlapRatio(candidate.tightBbox, neighbor.tightBbox) < 0.3) continue;

    const overlap = Math.max(
      0,
      Math.min(candidate.tightBbox.yMax, neighbor.tightBbox.yMax) -
        Math.max(candidate.tightBbox.yMin, neighbor.tightBbox.yMin)
    );
    if (overlap <= threshold) continue;
    penalty = Math.max(penalty, 3.5 + Math.min(6.0, (overlap - threshold) * 0.6));
  }
  return penalty;
}

function dbnetLineBoundaryIntrusionPenalty(candidate, allCandidates) {
  const profiles = candidate?.profiles || [];
  if (!candidate?.strokeIds?.length || !candidate.tightBbox) return 0;
  if (profiles.includes('dbnet-line')) return 0;
  if (!profiles.some((profile) => (
    profile === 'strict' ||
    profile === 'raw-row-line' ||
    profile === 'row-line' ||
    profile === 'loose'
  ))) {
    return 0;
  }

  const children = (allCandidates || []).filter((child) => (
    child !== candidate &&
    child?.profiles?.includes('dbnet-line') &&
    child.tightBbox &&
    strokeSetContains(candidate, child)
  ));
  for (const child of children) {
    const childIds = new Set(child.strokeIds || []);
    const extraStrokes = (candidate.strokes || []).filter((stroke) => !childIds.has(String(stroke.id)));
    if (extraStrokes.length === 0 || extraStrokes.length > 3) continue;
    if ((child.strokeIds || []).length < (candidate.strokeIds || []).length * 0.72) continue;
    if (bboxWidth(child.tightBbox) < bboxWidth(candidate.tightBbox) * 0.72) continue;
    if (candidate.tightBbox.yMax - child.tightBbox.yMax > 18) continue;

    const lowerEdgeIntrusion = extraStrokes.every((stroke) => {
      const box = stroke.canvasBbox;
      if (!box) return false;
      if (centerY(box) < child.tightBbox.yMax - 8) return false;
      if (horizontalOverlapRatio(box, child.tightBbox) < 0.08 && horizontalGap(box, child.tightBbox) > 42) {
        return false;
      }
      return bboxHeight(box) <= Math.max(42, bboxHeight(child.tightBbox) * 0.45);
    });
    if (lowerEdgeIntrusion) return 4.2;
  }
  return 0;
}

function dbnetBoundaryStrokeIntrusionPenalty(candidate, allCandidates, { medianHeight }) {
  const profiles = candidate?.profiles || [];
  if (!profiles.includes('dbnet-line') || !candidate?.strokeIds?.length || !candidate.tightBbox) {
    return 0;
  }

  for (const child of allCandidates || []) {
    if (child === candidate || !child?.tightBbox || !child?.strokeIds?.length) continue;
    if (!isPreferredChildLineCandidate(child)) continue;
    if (!strokeSetContains(candidate, child)) continue;
    if ((child.strokeIds || []).length < (candidate.strokeIds || []).length - 3) continue;
    if (bboxWidth(child.tightBbox) < bboxWidth(candidate.tightBbox) * 0.68) continue;

    const childIds = new Set(child.strokeIds || []);
    const extraStrokes = (candidate.strokes || []).filter((stroke) => !childIds.has(String(stroke.id)));
    if (extraStrokes.length === 0 || extraStrokes.length > 3) continue;

    const upperIntrusion = extraStrokes.every((stroke) => {
      const box = stroke.canvasBbox;
      if (!box) return false;
      if (centerY(box) >= child.tightBbox.yMin) return false;
      const gap = child.tightBbox.yMin - (box.yMax ?? child.tightBbox.yMin);
      if (gap > Math.max(24, (medianHeight || 1) * 0.8)) return false;
      if (bboxHeight(box) > Math.max(42, bboxHeight(child.tightBbox) * 0.45)) return false;
      if (horizontalOverlapRatio(box, child.tightBbox) < 0.08 && horizontalGap(box, child.tightBbox) > 42) {
        return false;
      }
      return true;
    });
    if (upperIntrusion) return 5.2;
  }
  return 0;
}

function isStructuralMathCandidate(candidate) {
  if (!candidate?.strokes?.length) return false;
  const rows = splitCandidateIntoRows(candidate, DEFAULT_CONFIG);
  return hasFractionLikeBridge(candidate) ||
    hasCompactStackedFractionColumn(candidate, rows) ||
    containsFractionBridge(candidate, rows) ||
    rows.some((row) => rowHasTallOperatorStroke(row, rowMedianHeight(row)));
}

function hasCompactFractionContainer(candidate, allCandidates) {
  if (candidate.profiles?.includes('fraction-stack-line')) return false;
  return (allCandidates || []).some((container) => {
    if (!container || container === candidate) return false;
    if (!strokeSetContains(container, candidate)) return false;
    if ((container.strokeIds || []).length <= (candidate.strokeIds || []).length) return false;
    if (
      bboxHeight(container.tightBbox) > 140 &&
      bboxWidth(candidate.tightBbox) > bboxWidth(container.tightBbox) * 0.75
    ) return false;
    if (bboxHeight(candidate.tightBbox) > bboxHeight(container.tightBbox) * 0.78) return false;
    return container.profiles?.includes('fraction-stack-line') || hasCompactFractionStructure(container);
  });
}

function hasCompactFractionStructure(candidate) {
  const rawRows = clusterStrokeRows(candidate, DEFAULT_CONFIG);
  if (rawRows.length <= 1 || rawRows.length > 3) return false;
  const medianHeight = median(candidate.strokes.map(strokeHeight)) || 1;
  const spread = bboxHeight(candidate.tightBbox) / medianHeight;
  if (spread > 5.2) {
    return spread <= 12 &&
      bboxHeight(candidate.tightBbox) <= 140 &&
      hasNearbyCompactFractionBridge(candidate, rawRows);
  }
  return hasProminentLocalFractionBridge(candidate, rawRows) ||
    hasInlineFractionBridge(candidate) ||
    hasCompactStackedFractionColumn(candidate, rawRows) ||
    hasNearbyCompactFractionBridge(candidate, rawRows);
}

function hasAttachedPlusMinusStroke(candidate, rows = null) {
  if (!candidate?.strokes?.length || !candidate.tightBbox) return false;
  const sortedRows = (rows || clusterStrokeRows(candidate, DEFAULT_CONFIG)).slice().sort(compareRows);
  if (sortedRows.length !== 2) return false;

  const upper = sortedRows[0];
  const lower = sortedRows[1];
  if ((lower.strokes || []).length !== 1) return false;
  const mark = lower.strokes[0];
  if (!isHorizontalStroke(mark)) return false;

  const markBox = mark.canvasBbox;
  const markWidth = bboxWidth(markBox);
  if (markWidth < 18 || markWidth > Math.max(90, bboxWidth(candidate.tightBbox) * 0.36)) return false;
  if (bboxHeight(markBox) > Math.max(14, rowMedianHeight(upper) * 0.38)) return false;

  const gap = verticalGap(upper.bbox, lower.bbox);
  if (gap > Math.max(18, rowMedianHeight(upper) * 0.45)) return false;
  if (horizontalOverlapRatio(markBox, upper.bbox) < 0.18) return false;

  const upperHorizontals = (upper.strokes || []).filter((stroke) => (
    stroke !== mark &&
    isHorizontalStroke(stroke) &&
    horizontalGap(stroke.canvasBbox, markBox) <= Math.max(55, markWidth * 1.2)
  ));
  return upperHorizontals.length > 0;
}

function inlineFragmentContinuityBonus(candidate, allCandidates = [], { medianHeight = 1 } = {}) {
  if (!candidate?.tightBbox || !candidate?.strokeIds?.length) return 0;
  if ((candidate.strokeIds || []).length < 5) return 0;
  const profiles = candidate.profiles || [];
  if (!profiles.some((profile) => (
    profile === 'row-line' ||
    profile === 'raw-row-line' ||
    profile === 'loose' ||
    profile === 'strict'
  ))) {
    return 0;
  }

  const children = (allCandidates || [])
    .filter((child) => (
      child !== candidate &&
      child?.tightBbox &&
      child.strokeIds?.length &&
      (child.strokeIds || []).length < (candidate.strokeIds || []).length &&
      strokeSetContains(candidate, child) &&
      child.profiles?.includes('strict')
    ))
    .sort((a, b) => centerX(a.tightBbox) - centerX(b.tightBbox));
  if (children.length < 2 || !childLinesCoverParent(candidate, children)) return 0;

  const broadEnough = bboxWidth(candidate.tightBbox) >= Math.max(85, (medianHeight || 1) * 2.2);
  if (!broadEnough) return 0;

  let alignedPairs = 0;
  for (let index = 0; index < children.length - 1; index += 1) {
    const left = children[index].tightBbox;
    const right = children[index + 1].tightBbox;
    const maxHeight = Math.max(1, bboxHeight(left), bboxHeight(right));
    const aligned = verticalOverlapRatio(left, right) >= 0.22 ||
      Math.abs(centerY(left) - centerY(right)) <= Math.max(18, maxHeight * 0.55);
    const close = horizontalGap(left, right) <= Math.max(90, maxHeight * 1.6);
    if (aligned && close) alignedPairs += 1;
  }

  if (alignedPairs === 0) return 0;
  return Math.min(6.0, 3.4 + alignedPairs * 1.2);
}

function hasProminentLocalFractionBridge(candidate, rows) {
  if (!candidate?.strokes?.length || !rows || rows.length < 2) return false;
  const candidateBox = computeTightBbox(candidate.strokes);
  const candidateWidth = candidateBox ? Math.max(1, bboxWidth(candidateBox)) : 1;

  for (const stroke of candidate.strokes) {
    const bar = stroke.canvasBbox;
    if (!bar || !isHorizontalStroke(stroke)) continue;
    const width = bboxWidth(bar);
    const height = Math.max(1, bboxHeight(bar));
    if (width / height < 3) continue;
    if (width < candidateWidth * 0.34) continue;

    const barY = centerY(bar);
    let bandAbove = false;
    let bandBelow = false;
    for (const row of rows) {
      if (horizontalOverlapRatio(bar, row.bbox) < 0.2) continue;
      if (centerY(row.bbox) < barY) bandAbove = true;
      if (centerY(row.bbox) > barY) bandBelow = true;
    }
    if (bandAbove && bandBelow) return true;
  }

  return false;
}

function hasInlineFractionBridge(candidate) {
  if (!candidate?.strokes?.length) return false;

  for (const stroke of candidate.strokes) {
    const bar = stroke.canvasBbox;
    if (!bar || !isHorizontalStroke(stroke)) continue;
    const width = bboxWidth(bar);
    const height = Math.max(1, bboxHeight(bar));
    if (width / height < 3) continue;
    if (width < 12) continue;

    const barY = centerY(bar);
    let above = false;
    let below = false;
    let nearestAboveGap = Infinity;
    let nearestBelowGap = Infinity;

    for (const other of candidate.strokes) {
      if (other === stroke || !other.canvasBbox) continue;
      const box = other.canvasBbox;
      if (horizontalOverlapRatio(bar, box) < 0.18) continue;
      const otherY = centerY(box);
      if (otherY < barY - height * 0.25) {
        above = true;
        nearestAboveGap = Math.min(nearestAboveGap, Math.max(0, bar.yMin - box.yMax));
      }
      if (otherY > barY + height * 0.25) {
        below = true;
        nearestBelowGap = Math.min(nearestBelowGap, Math.max(0, box.yMin - bar.yMax));
      }
    }

    if (!above || !below) continue;
    const aboveGap = Math.max(4, nearestAboveGap);
    const belowGap = Math.max(4, nearestBelowGap);
    if (Math.max(aboveGap, belowGap) / Math.min(aboveGap, belowGap) <= 3.5) return true;
  }

  return false;
}

export function computeTightBbox(strokes) {
  let xMin = Infinity;
  let yMin = Infinity;
  let xMax = -Infinity;
  let yMax = -Infinity;
  for (const stroke of strokes || []) {
    const box = stroke?.canvasBbox;
    if (!box) continue;
    xMin = Math.min(xMin, box.xMin);
    yMin = Math.min(yMin, box.yMin);
    xMax = Math.max(xMax, box.xMax);
    yMax = Math.max(yMax, box.yMax);
  }
  if (!Number.isFinite(xMin)) return null;
  return { xMin, yMin, xMax, yMax };
}

export function verticalOverlapRatio(a, b) {
  if (!a || !b) return 0;
  const overlap = Math.max(0, Math.min(a.yMax, b.yMax) - Math.max(a.yMin, b.yMin));
  const smaller = Math.min(bboxHeight(a), bboxHeight(b));
  return smaller > 0 ? overlap / smaller : 0;
}

export function horizontalOverlapRatio(a, b) {
  if (!a || !b) return 0;
  const overlap = Math.max(0, Math.min(a.xMax, b.xMax) - Math.max(a.xMin, b.xMin));
  const smaller = Math.min(bboxWidth(a), bboxWidth(b));
  return smaller > 0 ? overlap / smaller : 0;
}

export function clusterStrokeRows(candidateOrStrokes, config = DEFAULT_CONFIG) {
  const strokes = Array.isArray(candidateOrStrokes)
    ? candidateOrStrokes
    : candidateOrStrokes?.strokes || [];
  const usable = strokes.filter((stroke) => stroke?.canvasBbox);
  if (usable.length === 0) return [];

  const heights = usable.map(strokeHeight);
  const medianHeight = median(heights) || 1;
  const centerThreshold = Math.max(8, medianHeight * config.rowCenterThresholdRatio);
  const rows = [];

  for (const stroke of usable.slice().sort(compareStrokeCenterY)) {
    const box = stroke.canvasBbox;
    const center = centerY(box);
    let best = null;
    let bestDistance = Infinity;

    for (const row of rows) {
      const distance = Math.abs(center - centerY(row.bbox));
      if (
        (verticalOverlapRatio(box, row.bbox) >= 0.15 || distance <= centerThreshold) &&
        distance < bestDistance
      ) {
        best = row;
        bestDistance = distance;
      }
    }

    if (best) {
      best.bbox = bboxUnion(best.bbox, box);
      best.strokes.push(stroke);
    } else {
      rows.push({ bbox: { ...box }, strokes: [stroke], detections: [] });
    }
  }

  return mergeTinyUpperAttachmentRows(rows, medianHeight).sort(compareRows);
}

export function splitCandidateIntoRows(candidate, config = DEFAULT_CONFIG) {
  const rows = clusterStrokeRows(candidate, config);
  if (rows.length <= 1) return rows;
  return mergeStructuralRows(candidate, rows, config);
}

function mergeTinyUpperAttachmentRows(rows, medianHeight = 1) {
  const mergedRows = (rows || []).map((row) => ({
    ...row,
    bbox: row.bbox ? { ...row.bbox } : null,
    strokes: (row.strokes || []).slice()
  }));
  if (mergedRows.length < 2) return mergedRows;

  for (let index = 0; index < mergedRows.length; index += 1) {
    const row = mergedRows[index];
    if (!tinyRowCanAttachAbove(row, medianHeight)) continue;
    const target = mergedRows
      .filter((candidate, candidateIndex) => (
        candidateIndex !== index &&
        candidate?.bbox &&
        centerY(candidate.bbox) > centerY(row.bbox) &&
        (candidate.strokes || []).length >= 3 &&
        tinyUpperRowFitsCandidate(row, candidate, medianHeight)
      ))
      .sort((a, b) => (
        verticalGap(row.bbox, a.bbox) - verticalGap(row.bbox, b.bbox) ||
        Math.abs(centerX(row.bbox) - centerX(a.bbox)) - Math.abs(centerX(row.bbox) - centerX(b.bbox))
      ))[0];

    if (!target) continue;
    target.strokes = uniqueStrokes([...(target.strokes || []), ...(row.strokes || [])]);
    target.bbox = bboxUnion(target.bbox, row.bbox);
    row.strokes = [];
  }

  return mergedRows.filter((row) => row.strokes.length > 0);
}

function tinyRowCanAttachAbove(row, medianHeight = 1) {
  if (!row?.bbox || (row.strokes || []).length !== 1) return false;
  const width = bboxWidth(row.bbox);
  const height = bboxHeight(row.bbox);
  const maxSize = Math.max(12, medianHeight * 0.38);
  return width <= maxSize && height <= maxSize;
}

function tinyUpperRowFitsCandidate(row, candidate, medianHeight = 1) {
  const gap = verticalGap(row.bbox, candidate.bbox);
  if (gap > Math.max(14, medianHeight * 0.45)) return false;
  if (bboxWidth(candidate.bbox) < Math.max(80, bboxWidth(row.bbox) * 6)) return false;
  if (horizontalOverlapRatio(row.bbox, candidate.bbox) < 0.1 && horizontalGap(row.bbox, candidate.bbox) > Math.max(24, medianHeight * 0.7)) {
    return false;
  }
  return row.bbox.yMax <= candidate.bbox.yMin + Math.max(12, medianHeight * 0.32);
}

function emptySegmentation() {
  return {
    candidates: [],
    selected: [],
    partitions: {},
    parentCandidateId: null
  };
}

function filterStrokes(strokes, answerBox) {
  const usable = strokes.filter((stroke) => stroke?.canvasBbox);
  if (!answerBox) return usable;
  return usable.filter((stroke) => strokeBelongsToAnswerBox(stroke, answerBox));
}

export function strokeBelongsToAnswerBox(stroke, answerBox, options = {}) {
  const box = stroke?.canvasBbox;
  if (!box) return false;
  if (!answerBox) return true;
  if (!bboxesOverlap(box, answerBox)) return false;
  if (
    centerX(box) >= answerBox.xMin &&
    centerX(box) <= answerBox.xMax &&
    centerY(box) >= answerBox.yMin &&
    centerY(box) <= answerBox.yMax
  ) {
    return true;
  }

  const overlapWidth = Math.max(0, Math.min(box.xMax, answerBox.xMax) - Math.max(box.xMin, answerBox.xMin));
  const overlapHeight = Math.max(0, Math.min(box.yMax, answerBox.yMax) - Math.max(box.yMin, answerBox.yMin));
  const overlapArea = overlapWidth * overlapHeight;
  const strokeArea = Math.max(1, bboxWidth(box) * bboxHeight(box));
  const minOverlapRatio = Number.isFinite(Number(options.minOverlapRatio))
    ? Number(options.minOverlapRatio)
    : 0.35;
  return overlapArea / strokeArea >= minOverlapRatio;
}

function buildBaseGroups(strokes, config) {
  const temporal = mergeCatchmentBatches(buildTemporalBatches(strokes, config), config);
  return temporal.length > 0 ? temporal : [{ strokes: strokes.slice() }];
}

function buildTemporalBatches(strokes, config) {
  const ordered = strokes
    .filter((stroke) => stroke?.canvasBbox)
    .slice()
    .sort((a, b) => strokeTime(a) - strokeTime(b));
  const batches = [];
  let current = null;
  let previousEnd = null;

  for (const stroke of ordered) {
    const start = strokeTime(stroke);
    if (!current || previousEnd === null || start - previousEnd > config.idleDelayMs) {
      current = { strokes: [] };
      batches.push(current);
    }
    current.strokes.push(stroke);
    previousEnd = Number.isFinite(Number(stroke.endTime)) ? Number(stroke.endTime) : start;
  }

  return batches;
}

function mergeCatchmentBatches(batches, config) {
  const groups = batches.map((batch) => ({ strokes: batch.strokes.slice() }));
  let changed = true;
  while (changed) {
    changed = false;
    outer:
    for (let i = 0; i < groups.length; i += 1) {
      for (let j = i + 1; j < groups.length; j += 1) {
        if (boxesCanMerge(groups[i], groups[j], config)) {
          groups[i].strokes = uniqueStrokes(groups[i].strokes.concat(groups[j].strokes));
          groups.splice(j, 1);
          changed = true;
          break outer;
        }
      }
    }
  }
  return groups;
}

function boxesCanMerge(a, b, config) {
  const tightA = computeTightBbox(a.strokes);
  const tightB = computeTightBbox(b.strokes);
  const catchA = computeExpandedBbox(a.strokes, 'temporal', config);
  const catchB = computeExpandedBbox(b.strokes, 'temporal', config);
  return bboxesOverlap(tightA, catchB) || bboxesOverlap(tightB, catchA);
}

function buildOverlapGroups(strokes, profile, config) {
  const groups = strokes
    .filter((stroke) => stroke?.canvasBbox)
    .map((stroke) => candidateFromStrokes([stroke], { profile, config }));

  let changed = true;
  while (changed) {
    changed = false;
    outer:
    for (let i = 0; i < groups.length; i += 1) {
      for (let j = i + 1; j < groups.length; j += 1) {
        if (groupsCanMerge(groups[i], groups[j], profile, config)) {
          groups[i] = candidateFromStrokes(
            groups[i].strokes.concat(groups[j].strokes),
            { profile, config }
          );
          groups.splice(j, 1);
          changed = true;
          break outer;
        }
      }
    }
  }
  return groups;
}

function buildProjectionLineGroups(strokes) {
  const usable = (strokes || [])
    .filter((stroke) => stroke?.canvasBbox)
    .slice()
    .sort((a, b) => (
      a.canvasBbox.yMin - b.canvasBbox.yMin ||
      a.canvasBbox.xMin - b.canvasBbox.xMin
    ));
  if (usable.length === 0) return [];

  const medianHeight = median(usable.map(strokeHeight)) || 1;
  const splitGap = Math.max(28, medianHeight * 0.72);
  const groups = [];
  let current = [];
  let currentBox = null;

  for (const stroke of usable) {
    const box = stroke.canvasBbox;
    const gap = currentBox ? box.yMin - currentBox.yMax : 0;
    if (current.length > 0 && gap > splitGap) {
      groups.push({ strokes: current });
      current = [];
      currentBox = null;
    }
    current.push(stroke);
    currentBox = currentBox ? bboxUnion(currentBox, box) : { ...box };
  }
  if (current.length > 0) groups.push({ strokes: current });

  return groups.filter((group) => group.strokes.length > 1);
}

function buildCompactFractionStackGroups(rows, config) {
  const sortedRows = (rows || []).slice().sort(compareRows);
  const groups = [];
  for (let index = 0; index < sortedRows.length - 1; index += 1) {
    for (let span = 2; span <= 3 && index + span <= sortedRows.length; span += 1) {
      const slice = sortedRows.slice(index, index + span);
      const strokes = uniqueStrokes(slice.flatMap((row) => row.strokes || []));
      if (strokes.length <= 1) continue;
      const candidate = candidateFromStrokes(strokes, {
        profile: 'fraction-stack-probe',
        config
      });
      if (
        slice.length > 1 &&
        bboxWidth(candidate.tightBbox) > 220 &&
        (
          rowLooksLikeEquationWithEquals(slice[slice.length - 1]) ||
          rowLooksLikeOperationAnnotation(slice[slice.length - 1])
        ) &&
        !hasWideFractionBar(candidate)
      ) {
        continue;
      }
      if (bboxHeight(candidate.tightBbox) > 150) continue;
      if (rowsHaveIndependentLowerContinuation(slice, candidate, { broadContinuationMin: 0 })) continue;
      if (!hasNearbyCompactFractionBridge(candidate, clusterStrokeRows(candidate, config))) continue;
      groups.push({ strokes });
    }
  }
  return groups;
}

function buildTallFractionStackGroups(rows, config) {
  const sortedRows = (rows || []).slice().sort(compareRows);
  const groups = [];
  for (let index = 0; index < sortedRows.length - 1; index += 1) {
    for (let span = 2; span <= 5 && index + span <= sortedRows.length; span += 1) {
      const slice = sortedRows.slice(index, index + span);
      const strokes = uniqueStrokes(slice.flatMap((row) => row.strokes || []));
      if (strokes.length <= 2) continue;
      const candidate = candidateFromStrokes(strokes, {
        profile: 'fraction-stack-probe',
        config
      });
      const height = bboxHeight(candidate.tightBbox);
      if (height <= 150 || height > 285) continue;
      const candidateRows = clusterStrokeRows(candidate, config);
      const wideSupportedFraction = hasWideFractionBar(candidate) &&
        hasProminentLocalFractionBridge(candidate, candidateRows);
      if (!wideSupportedFraction && !hasProminentLocalFractionBridge(candidate, candidateRows)) {
        continue;
      }
      if (
        rowsHaveIndependentLowerContinuation(slice, candidate, { broadContinuationMin: 360 }) &&
        !wideSupportedFraction
      ) {
        continue;
      }
      if (bottomRowFallsBelowFractionStack(slice, candidate)) continue;
      groups.push({ strokes });
    }
  }
  return groups;
}

function buildSuperscriptLineGroups(rows, config) {
  const sorted = (rows || [])
    .filter((row) => row?.bbox && row.strokes?.length)
    .slice()
    .sort(compareRows);
  if (sorted.length < 2) return [];

  const groups = [];
  const seen = new Set();
  for (const base of sorted) {
    const attached = sorted.filter((row) => row !== base && superscriptRowsCanAttach(base, row, sorted, config));
    if (!attached.length) continue;
    const strokes = uniqueStrokes([
      ...(base.strokes || []),
      ...attached.flatMap((row) => row.strokes || [])
    ]);
    if (strokes.length <= base.strokes.length) continue;
    const key = strokeSetKeyFor(strokes);
    if (seen.has(key)) continue;
    seen.add(key);
    groups.push({
      strokes,
      baseRow: base,
      exponentRows: attached
    });
  }
  return groups;
}

function hasSuperscriptStructure(rows, medianHeight = 1) {
  const sorted = (rows || []).filter((row) => row?.bbox && row.strokes?.length);
  if (sorted.length < 2) return false;
  const config = {
    ...DEFAULT_CONFIG,
    rowCenterThresholdRatio: Math.max(
      DEFAULT_CONFIG.rowCenterThresholdRatio,
      Number(medianHeight) > 0 ? 0.82 : DEFAULT_CONFIG.rowCenterThresholdRatio
    )
  };
  return sorted.some((base) => sorted.some((row) => (
    row !== base && superscriptRowsCanAttach(base, row, sorted, config)
  )));
}

function superscriptRowsCanAttach(base, exponent, rows, config = DEFAULT_CONFIG) {
  if (!base?.bbox || !exponent?.bbox) return false;
  if ((base.strokes || []).length > 5) return false;
  if (!rowHasSuperscriptBaseInk(base)) return false;
  if (centerY(exponent.bbox) >= centerY(base.bbox)) return false;

  const baseHeight = bboxHeight(base.bbox);
  const exponentHeight = bboxHeight(exponent.bbox);
  const medianHeight = median([
    ...(base.strokes || []).map(strokeHeight),
    ...(exponent.strokes || []).map(strokeHeight)
  ]) || Math.max(baseHeight, exponentHeight, 1);
  const gap = verticalGap(base.bbox, exponent.bbox);
  const centerDeltaY = centerY(base.bbox) - centerY(exponent.bbox);
  const closeVertically = gap <= Math.max(24, medianHeight * 1.15) &&
    centerDeltaY <= Math.max(62, medianHeight * 2.65);
  if (!closeVertically) return false;

  const baseWidth = bboxWidth(base.bbox);
  const exponentWidth = bboxWidth(exponent.bbox);
  const rightOfBaseStart = exponent.bbox.xMax >= base.bbox.xMin + Math.min(34, baseWidth * 0.28);
  const notFarLeft = exponent.bbox.xMin >= base.bbox.xMin - Math.max(18, medianHeight * 0.8);
  const closeHorizontally = horizontalGap(base.bbox, exponent.bbox) <= Math.max(175, medianHeight * 6, baseWidth * 1.4);
  if (!rightOfBaseStart || !notFarLeft || !closeHorizontally) return false;

  const baseLooksCompact = baseWidth <= Math.min(140, Math.max(100, medianHeight * 4.2));
  const exponentLooksUpperRight = centerX(exponent.bbox) >= base.bbox.xMin + Math.min(22, baseWidth * 0.2);
  const baseLooksLikeEquation = rowLooksLikeEquationWithEquals(base);
  if ((!baseLooksCompact && !baseLooksLikeEquation) || !exponentLooksUpperRight) return false;
  if (!baseLooksCompact && gap > Math.max(16, medianHeight * 0.45)) return false;

  if (
    exponentWidth > Math.max(620, baseWidth * 3.4) &&
    exponent.bbox.xMin < base.bbox.xMin + Math.max(28, baseWidth * 0.18)
  ) {
    return false;
  }

  const interveningRows = (rows || []).filter((row) => (
    row !== base &&
    row !== exponent &&
    row?.bbox &&
    centerY(row.bbox) > centerY(exponent.bbox) &&
    centerY(row.bbox) < centerY(base.bbox) &&
    horizontalOverlapRatio(row.bbox, base.bbox) >= 0.2
  ));
  return interveningRows.length === 0;
}

function rowHasSuperscriptBaseInk(row) {
  const strokes = (row?.strokes || []).filter((stroke) => stroke?.canvasBbox);
  if (!strokes.length) return false;
  const medianHeight = rowMedianHeight(row) || 1;
  return strokes.some((stroke) => {
    const box = stroke.canvasBbox;
    if (isHorizontalStroke(stroke)) return false;
    const height = bboxHeight(box);
    const width = bboxWidth(box);
    return height >= Math.max(12, medianHeight * 0.45) || height >= width * 0.55;
  });
}

function superscriptSplitChildPenalty(candidate, allCandidates = []) {
  const profiles = candidate?.profiles || [];
  if (!profiles.some((profile) => (
    profile === 'row-line' ||
    profile === 'raw-row-line' ||
    profile === 'strict' ||
    profile === 'loose'
  ))) {
    return 0;
  }
  const parent = (allCandidates || []).find((other) => (
    other !== candidate &&
    other?.profiles?.includes('superscript-line') &&
    (other.strokeIds || []).length > (candidate.strokeIds || []).length &&
    strokeSetContains(other, candidate) &&
    hasSuperscriptStructure(clusterStrokeRows(other, DEFAULT_CONFIG), rowMedianHeight(candidate) || 1)
  ));
  return parent ? 4.8 : 0;
}

function bottomRowFallsBelowFractionStack(rows, candidate) {
  const sortedRows = (rows || []).slice().sort(compareRows);
  if (sortedRows.length < 3) return false;

  const bottom = sortedRows[sortedRows.length - 1];
  const bottomStrokeIds = new Set((bottom.strokes || []).map((stroke) => String(stroke.id)));
  const bars = (candidate?.strokes || [])
    .filter((stroke) => (
      stroke?.canvasBbox &&
      isHorizontalStroke(stroke) &&
      !bottomStrokeIds.has(String(stroke.id))
    ))
    .map((stroke) => stroke.canvasBbox)
    .filter((box) => centerY(box) < centerY(bottom.bbox))
    .sort((a, b) => centerY(b) - centerY(a));
  const bar = bars[0];
  if (!bar) return false;

  const bottomGap = Math.max(0, bottom.bbox.yMin - bar.yMax);
  const centerGap = centerY(bottom.bbox) - centerY(bar);
  const bottomMedian = rowMedianHeight(bottom) || bboxHeight(bottom.bbox);
  return bottomGap > Math.max(42, bottomMedian * 0.85) &&
    centerGap > Math.max(70, bottomMedian * 1.45);
}

function rowsHaveIndependentLowerContinuation(rows, candidate, options = {}) {
  const sortedRows = (rows || []).slice().sort(compareRows);
  for (let index = 0; index < sortedRows.length - 1; index += 1) {
    if (looksLikeIndependentLowerContinuation(
      sortedRows[index],
      sortedRows[index + 1],
      candidate,
      options
    )) {
      return true;
    }
  }
  return false;
}

function buildCenterGapLineGroups(row, config) {
  const strokes = (row?.strokes || [])
    .filter((stroke) => stroke?.canvasBbox)
    .slice()
    .sort(compareStrokeCenterY);
  if (strokes.length < 8) return [];

  const parentBox = computeTightBbox(strokes);
  if (!parentBox || bboxHeight(parentBox) < 145) return [];
  const medianHeight = median(strokes.map(strokeHeight)) || 1;
  const minCenterGap = Math.max(14, medianHeight * 0.2);
  let best = null;

  for (let index = 0; index < strokes.length - 1; index += 1) {
    const upper = strokes.slice(0, index + 1);
    const lower = strokes.slice(index + 1);
    if (upper.length < 3 || lower.length < 3) continue;

    const gap = centerY(strokes[index + 1].canvasBbox) - centerY(strokes[index].canvasBbox);
    if (gap < minCenterGap) continue;

    const upperBox = computeTightBbox(upper);
    const lowerBox = computeTightBbox(lower);
    if (!upperBox || !lowerBox) continue;
    if (bboxWidth(upperBox) < Math.max(120, bboxWidth(parentBox) * 0.2)) continue;
    if (bboxWidth(lowerBox) < Math.max(120, bboxWidth(parentBox) * 0.2)) continue;

    const verticalOverlap = Math.max(0, Math.min(upperBox.yMax, lowerBox.yMax) - Math.max(upperBox.yMin, lowerBox.yMin));
    if (verticalOverlap > Math.max(8, medianHeight * 0.18)) continue;

    const score = gap - verticalOverlap * 2;
    if (!best || score > best.score) {
      best = { upper, lower, score };
    }
  }

  if (!best) return [];
  const bands = [
    { strokes: best.upper, bbox: computeTightBbox(best.upper), detections: [] },
    { strokes: best.lower, bbox: computeTightBbox(best.lower), detections: [] }
  ];
  return bands;
}

function groupsCanMerge(a, b, profile, config) {
  if (!bboxesOverlap(a.expandedBbox, b.expandedBbox)) return false;
  if (profile === 'strict') {
    return verticalOverlapRatio(a.tightBbox, b.tightBbox) >= config.strictMinVerticalOverlapRatio;
  }
  return true;
}

function computeExpandedBbox(strokes, profile, config) {
  const box = computeTightBbox(strokes);
  if (!box) return null;
  const width = bboxWidth(box);
  const height = bboxHeight(box);
  let hPad = config.horizontalPadding;
  let vPad = config.verticalPadding;

  if (profile === 'loose') {
    hPad = Math.max(hPad, width * 0.35);
    vPad = Math.max(vPad, height * 0.10);
  } else if (profile === 'strict') {
    hPad = Math.max(8, width * 0.18);
    vPad = Math.max(4, height * 0.05);
  } else {
    vPad = Math.max(vPad, Math.min(40, height * 0.5));
  }

  return {
    xMin: box.xMin - hPad,
    yMin: box.yMin - vPad,
    xMax: box.xMax + hPad,
    yMax: box.yMax + vPad
  };
}

function candidateFromStrokes(strokes, { profile, profiles, config, ...extra } = {}) {
  const usable = uniqueStrokes(strokes).filter((stroke) => stroke?.canvasBbox);
  const tightBbox = computeTightBbox(usable);
  const strokeIds = usable.map((stroke) => String(stroke.id));
  const key = strokeSetKeyFor(usable);
  const candidateId = `${profile || 'candidate'}_${key}`;
  return {
    id: candidateId,
    candidateId,
    profile: profile || 'candidate',
    profiles: profiles || [profile || 'candidate'],
    strokeSetKey: key,
    strokeIds,
    strokes: usable,
    tightBbox,
    expandedBbox: computeExpandedBbox(usable, profile || 'candidate', config || DEFAULT_CONFIG),
    conflicts: [],
    sources: extra.sources || [profile || 'candidate'],
    ...extra
  };
}

function splitCandidateByDetections(candidate, detectedBands, config) {
  const chosen = chooseLineBands(candidate, detectedBands, config);
  if (chosen.length <= 1) return [candidate];
  if (chosen.length <= 2 && containsFractionBridge(candidate, chosen)) return [candidate];

  const bands = mergeStructuralRows(candidate, assignStrokesToBands(candidate, chosen), config);
  if (bands.length <= 1) return [candidate];
  if (bands.length <= 2 && containsFractionBridge(candidate, bands)) return [candidate];

  const lines = bands
    .map((band) => candidateFromStrokes(band.strokes, { profile: 'dbnet-line', config }))
    .filter((line) => line.strokes.length > 0);

  return lines.length > 1 ? sortCandidates(lines) : [candidate];
}

function clusterDetections(detections, minVerticalOverlapRatio) {
  const bands = (detections || [])
    .filter((item) => item?.bbox && item.bbox.yMax > item.bbox.yMin)
    .map((item) => ({ bbox: { ...item.bbox }, detections: [item], strokes: [] }))
    .sort(compareRows);

  let changed = true;
  while (changed) {
    changed = false;
    outer:
    for (let i = 0; i < bands.length; i += 1) {
      for (let j = i + 1; j < bands.length; j += 1) {
        if (verticalOverlapRatio(bands[i].bbox, bands[j].bbox) >= minVerticalOverlapRatio) {
          bands[i].bbox = bboxUnion(bands[i].bbox, bands[j].bbox);
          bands[i].detections = bands[i].detections.concat(bands[j].detections);
          bands.splice(j, 1);
          changed = true;
          break outer;
        }
      }
    }
  }
  return bands.sort(compareRows);
}

function chooseLineBands(candidate, detectedBands, config) {
  const strokeRows = splitCandidateIntoRows(candidate, config);
  if (strokeRows.length > detectedBands.length && strokeRows.length > 1) {
    if (detectedBandsOnlyExpandStructuralFractions(detectedBands, strokeRows)) return detectedBands;
    return strokeRows;
  }
  if (
    strokeRows.length === detectedBands.length &&
    strokeRows.length > 1 &&
    bandsHaveAmbiguousOverlap(detectedBands)
  ) {
    return strokeRows;
  }
  if (
    strokeRows.length > 1 &&
    detectedBands.length > strokeRows.length &&
    bandsHaveAmbiguousOverlap(detectedBands)
  ) {
    return strokeRows;
  }
  return detectedBands;
}

function assignStrokesToBands(candidate, bands) {
  const assigned = bands.map((band) => ({
    bbox: { ...band.bbox },
    detections: (band.detections || []).slice(),
    strokes: []
  }));

  for (const stroke of candidate.strokes || []) {
    const box = stroke.canvasBbox;
    if (!box || assigned.length === 0) continue;
    let bestIndex = 0;
    let bestOverlap = -1;
    let bestDistance = Infinity;

    for (let i = 0; i < assigned.length; i += 1) {
      const band = assigned[i].bbox;
      const overlap = Math.max(0, Math.min(box.yMax, band.yMax) - Math.max(box.yMin, band.yMin));
      const distance = Math.abs(centerY(box) - centerY(band));
      if (overlap > bestOverlap || (overlap === bestOverlap && distance < bestDistance)) {
        bestIndex = i;
        bestOverlap = overlap;
        bestDistance = distance;
      }
    }
    assigned[bestIndex].strokes.push(stroke);
  }

  return assigned.filter((band) => band.strokes.length > 0);
}

function mergeStructuralRows(candidate, rows, config) {
  let mergedRows = rows
    .filter((row) => row.strokes?.length)
    .map((row) => ({
      bbox: { ...row.bbox },
      strokes: uniqueStrokes(row.strokes),
      detections: (row.detections || []).slice()
    }))
    .sort(compareRows);

  mergedRows = mergeTouchingLocalStacks(mergedRows, candidate.tightBbox);

  let changed = true;
  while (changed) {
    changed = false;
    let best = null;
    for (let i = 0; i < mergedRows.length; i += 1) {
      for (let j = 0; j < mergedRows.length; j += 1) {
        if (i === j) continue;
        const gap = verticalGap(mergedRows[i].bbox, mergedRows[j].bbox);
        if (hasCloserUpperNeighbor(i, j, mergedRows, gap)) continue;
        if (!isStructuralAttachment(mergedRows[i], mergedRows[j], candidate, config)) continue;
        if (!best || gap < best.gap) best = { child: i, parent: j, gap };
      }
    }
    if (best) {
      mergeRows(mergedRows[best.parent], mergedRows[best.child]);
      mergedRows.splice(best.child, 1);
      changed = true;
    }
  }

  return mergedRows.sort(compareRows);
}

function mergeTouchingLocalStacks(rows, candidateBox) {
  const candidateWidth = candidateBox ? Math.max(1, bboxWidth(candidateBox)) : 1;
  let changed = true;
  while (changed) {
    changed = false;
    rows.sort(compareRows);
    for (let i = 0; i < rows.length - 1; i += 1) {
      const upper = rows[i];
      const lower = rows[i + 1];
      const gap = verticalGap(upper.bbox, lower.bbox);
      const verticalTouch = gap <= 6 || verticalOverlapRatio(upper.bbox, lower.bbox) >= 0.08;
      const aligned = horizontalOverlapRatio(upper.bbox, lower.bbox) >= 0.45;
      const upperWidth = bboxWidth(upper.bbox);
      const lowerWidth = bboxWidth(lower.bbox);
      const local = upperWidth <= candidateWidth * 0.55 && lowerWidth <= candidateWidth * 0.55;
      const widthRatio = Math.max(upperWidth, lowerWidth) / Math.max(1, Math.min(upperWidth, lowerWidth));
      const comparableWidths = widthRatio <= 1.55 ||
        (upperWidth <= candidateWidth * 0.35 && lowerWidth <= candidateWidth * 0.35);
      const compactLocal = upperWidth <= candidateWidth * 0.35 && lowerWidth <= candidateWidth * 0.35;
      const numeratorStack = verticalTouch &&
        horizontalOverlapRatio(upper.bbox, lower.bbox) >= 0.85 &&
        upperWidth <= lowerWidth * 0.58 &&
        upperWidth <= candidateWidth * 0.42;

      if (looksLikeIndependentLowerContinuation(upper, lower, { tightBbox: candidateBox })) {
        continue;
      }
      if (
        upperWidth <= lowerWidth * 0.62 &&
        rowHasNearbyUpperFractionBar(upper, rows.slice(0, i), candidateBox)
      ) {
        continue;
      }

      if (
        !numeratorStack &&
        (!verticalTouch || !aligned || !local || !comparableWidths || !compactLocal)
      ) {
        continue;
      }
      mergeRows(upper, lower);
      rows.splice(i + 1, 1);
      changed = true;
      break;
    }
  }
  return rows;
}

function rowHasNearbyUpperFractionBar(row, previousRows, candidateBox) {
  if (!row?.bbox || !(previousRows || []).length) return false;
  const candidateWidth = candidateBox ? Math.max(1, bboxWidth(candidateBox)) : Math.max(1, bboxWidth(row.bbox));
  const rowMedian = rowMedianHeight(row) || 1;
  for (const previous of previousRows || []) {
    if (!previous?.bbox || centerY(previous.bbox) >= centerY(row.bbox)) continue;
    const gap = verticalGap(previous.bbox, row.bbox);
    if (gap > Math.max(18, rowMedian * 0.75)) continue;
    for (const stroke of previous.strokes || []) {
      const bar = stroke.canvasBbox;
      if (!bar || !isHorizontalStroke(stroke)) continue;
      const width = bboxWidth(bar);
      if (width < Math.max(candidateWidth * 0.35, bboxWidth(row.bbox) * 2.4)) continue;
      if (centerY(bar) > centerY(row.bbox)) continue;
      if (horizontalOverlapRatio(bar, row.bbox) < 0.2) continue;
      return true;
    }
  }
  return false;
}

function isStructuralAttachment(child, parent, candidate) {
  if (!child?.bbox || !parent?.bbox || !child.strokes?.length || !parent.strokes?.length) {
    return false;
  }

  const childMedian = rowMedianHeight(child);
  const parentMedian = rowMedianHeight(parent);
  const childWidth = bboxWidth(child.bbox);
  const parentWidth = bboxWidth(parent.bbox);
  const childHeight = bboxHeight(child.bbox);
  const parentHeight = bboxHeight(parent.bbox);
  const gap = verticalGap(child.bbox, parent.bbox);
  const closeEnough = gap <= Math.max(26, parentMedian * 0.9, childMedian * 1.2);
  if (!closeEnough) return false;

  const childCenter = centerY(child.bbox);
  const parentCenter = centerY(parent.bbox);
  const verticallyOffset = Math.abs(childCenter - parentCenter) >= Math.min(childMedian, parentMedian) * 0.35;
  if (!verticallyOffset) return false;

  const horizontalOverlap = horizontalOverlapRatio(child.bbox, parent.bbox);
  const supportAllowance = Math.max(18, parentMedian * 0.8, childMedian * 1.2);
  const localSupport = rowHasLocalSupport(child, parent, supportAllowance) || horizontalOverlap >= 0.5;

  if (looksLikeIndependentLowerContinuation(child, parent, candidate)) {
    return false;
  }

  const decoration = childCenter > parentCenter &&
    mostlyHorizontal(child) &&
    childMedian <= parentMedian * 0.55 &&
    childHeight <= parentHeight * 0.45 &&
    gap <= Math.max(14, parentMedian * 0.55) &&
    horizontalOverlap >= 0.45;
  if (decoration) return true;

  if (localSupport) {
    const pairCandidate = candidateFromStrokes(uniqueStrokes(child.strokes.concat(parent.strokes)), {
      profile: 'structural-pair',
      config: DEFAULT_CONFIG
    });
    const narrower = Math.min(childWidth, parentWidth);
    const wider = Math.max(childWidth, parentWidth);
    const fractionWidthAsymmetry = narrower <= wider * 0.62;
    if (
      containsFractionBridge(pairCandidate, [child, parent]) &&
      (fractionWidthAsymmetry || bridgeBarLivesInLowerRow(child, parent, pairCandidate))
    ) {
      return true;
    }
  }

  const aboveAttachment = childCenter < parentCenter;
  const sparse = child.strokes.length <= Math.max(2, Math.floor(parent.strokes.length * 0.45));
  const verySparse = child.strokes.length <= Math.max(2, Math.floor(parent.strokes.length * 0.3));
  const compact = childWidth <= parentWidth * (aboveAttachment ? 0.55 : 0.35);
  const physicallySmall = childHeight <= parentHeight * 0.55 ||
    (childMedian <= parentMedian * 0.75 && childHeight <= parentHeight * 0.85);
  const lowerLimitLike = !aboveAttachment &&
    child.strokes.length <= 4 &&
    childWidth <= parentWidth * 0.32 &&
    rowHasTallOperatorStroke(parent, parentMedian);
  const numeratorLike = aboveAttachment &&
    child.strokes.length <= 4 &&
    childWidth <= parentWidth * 0.45 &&
    rowHasLocalHorizontalBridge(parent, child);

  return localSupport && (
    (aboveAttachment && (sparse || numeratorLike) && physicallySmall) ||
    (!aboveAttachment && (verySparse || lowerLimitLike) && compact && physicallySmall)
  );
}

function looksLikeIndependentLowerContinuation(child, parent, candidate, options = {}) {
  const upper = centerY(child.bbox) <= centerY(parent.bbox) ? child : parent;
  const lower = upper === child ? parent : child;

  const lowerWidth = bboxWidth(lower.bbox);
  const upperWidth = bboxWidth(upper.bbox);
  const lowerHeight = bboxHeight(lower.bbox);
  const upperHeight = bboxHeight(upper.bbox);
  const lowerMedian = rowMedianHeight(lower);
  const upperMedian = rowMedianHeight(upper);
  const candidateWidth = candidate?.tightBbox ? bboxWidth(candidate.tightBbox) : Math.max(lowerWidth, upperWidth);
  const broadContinuationMin = Number.isFinite(Number(options.broadContinuationMin))
    ? Number(options.broadContinuationMin)
    : 520;
  const broadContinuationWidth = Math.max(broadContinuationMin, candidateWidth * 0.72);

  return lowerWidth >= upperWidth * 1.35 &&
    lowerWidth >= broadContinuationWidth &&
    (lower.strokes || []).length >= Math.max(5, Math.floor((upper.strokes || []).length * 0.75)) &&
    lowerHeight >= upperHeight * 0.42 &&
    lowerMedian >= upperMedian * 0.58;
}

function containsFractionBridge(candidate, bands) {
  if (!candidate?.strokes?.length || !bands || bands.length < 2) return false;
  const candidateBox = computeTightBbox(candidate.strokes);
  const candidateWidth = candidateBox ? Math.max(1, bboxWidth(candidateBox)) : 1;

  for (const stroke of candidate.strokes) {
    const bar = stroke.canvasBbox;
    if (!bar) continue;
    const width = bboxWidth(bar);
    const height = Math.max(1, bboxHeight(bar));
    if (width / height < 3) continue;
    if (width < candidateWidth * 0.22) continue;

    const barY = centerY(bar);
    let inkAbove = false;
    let inkBelow = false;
    let nearestAboveGap = Infinity;
    let nearestBelowGap = Infinity;

    for (const other of candidate.strokes) {
      if (other === stroke || !other.canvasBbox) continue;
      const box = other.canvasBbox;
      if (horizontalOverlapRatio(bar, box) < 0.2) continue;
      const strokeY = centerY(box);
      if (strokeY < barY - height * 0.25) {
        inkAbove = true;
        nearestAboveGap = Math.min(nearestAboveGap, Math.max(0, bar.yMin - box.yMax));
      }
      if (strokeY > barY + height * 0.25) {
        inkBelow = true;
        nearestBelowGap = Math.min(nearestBelowGap, Math.max(0, box.yMin - bar.yMax));
      }
    }
    if (!inkAbove || !inkBelow) continue;

    const aboveGap = Math.max(6, nearestAboveGap);
    const belowGap = Math.max(6, nearestBelowGap);
    if (Math.max(aboveGap, belowGap) / Math.min(aboveGap, belowGap) > 3.25) continue;
    if (belowGap > Math.max(32, aboveGap * 3 + 8)) continue;

    let bandAbove = false;
    let bandBelow = false;
    let hasDistantBand = false;
    const distantThreshold = Math.max(36, height * 8);
    for (const band of bands) {
      const box = band.bbox;
      if (horizontalOverlapRatio(bar, box) < 0.2) continue;
      if (centerY(box) < barY) bandAbove = true;
      if (centerY(box) > barY) bandBelow = true;
      if (verticalGap(bar, box) > distantThreshold) hasDistantBand = true;
    }
    if (bandAbove && bandBelow && !hasDistantBand) return true;
  }

  return false;
}

function bridgeBarLivesInLowerRow(rowA, rowB, candidate) {
  const upper = centerY(rowA.bbox) <= centerY(rowB.bbox) ? rowA : rowB;
  const lower = upper === rowA ? rowB : rowA;
  const candidateBox = computeTightBbox(candidate.strokes);
  const candidateWidth = candidateBox ? Math.max(1, bboxWidth(candidateBox)) : 1;

  return (lower.strokes || []).some((stroke) => {
    const bar = stroke.canvasBbox;
    if (!bar || !isHorizontalStroke(stroke)) return false;
    const width = bboxWidth(bar);
    const height = Math.max(1, bboxHeight(bar));
    if (width < candidateWidth * 0.22) return false;
    if (centerY(upper.bbox) >= centerY(bar)) return false;
    if (centerY(lower.bbox) <= centerY(bar) + height * 0.25) return false;
    return horizontalOverlapRatio(bar, upper.bbox) >= 0.2;
  });
}

function hasFractionLikeBridge(candidate) {
  if (!candidate?.strokes?.length) return false;
  const candidateBox = computeTightBbox(candidate.strokes);
  const candidateWidth = candidateBox ? Math.max(1, bboxWidth(candidateBox)) : 1;

  for (const stroke of candidate.strokes) {
    const bar = stroke.canvasBbox;
    if (!bar) continue;
    const width = bboxWidth(bar);
    const height = Math.max(1, bboxHeight(bar));
    if (width / height < 3) continue;
    if (width < candidateWidth * 0.18) continue;

    const barY = centerY(bar);
    let inkAbove = false;
    let inkBelow = false;
    let nearestAboveGap = Infinity;
    let nearestBelowGap = Infinity;

    for (const other of candidate.strokes) {
      if (other === stroke || !other.canvasBbox) continue;
      const box = other.canvasBbox;
      if (horizontalOverlapRatio(bar, box) < 0.16) continue;
      const strokeY = centerY(box);
      if (strokeY < barY - height * 0.25) {
        inkAbove = true;
        nearestAboveGap = Math.min(nearestAboveGap, Math.max(0, bar.yMin - box.yMax));
      }
      if (strokeY > barY + height * 0.25) {
        inkBelow = true;
        nearestBelowGap = Math.min(nearestBelowGap, Math.max(0, box.yMin - bar.yMax));
      }
    }

    if (!inkAbove || !inkBelow) continue;
    const aboveGap = Math.max(6, nearestAboveGap);
    const belowGap = Math.max(6, nearestBelowGap);
    if (Math.max(aboveGap, belowGap) / Math.min(aboveGap, belowGap) <= 3.25) {
      return true;
    }
  }

  return false;
}

function detectedBandsOnlyExpandStructuralFractions(detectedBands, strokeRows) {
  if (!detectedBands || detectedBands.length <= 1) return false;
  if (!strokeRows || strokeRows.length <= detectedBands.length) return false;
  if (bandsHaveAmbiguousOverlap(detectedBands)) return false;

  let foundStructuralExpansion = false;
  for (const band of detectedBands) {
    const containedRows = strokeRowsInsideBand(strokeRows, band);
    if (containedRows.length <= 1) continue;
    const containedStrokes = uniqueStrokes(containedRows.flatMap((row) => row.strokes || []));
    if (!containsFractionBridge(
      candidateFromStrokes(containedStrokes, { profile: 'detected-fraction', config: DEFAULT_CONFIG }),
      containedRows
    )) {
      return false;
    }
    foundStructuralExpansion = true;
  }
  return foundStructuralExpansion;
}

function strokeRowsInsideBand(strokeRows, band) {
  const box = band.bbox;
  return strokeRows.filter((row) => {
    const rowBox = row.bbox;
    const center = centerY(rowBox);
    return verticalOverlapRatio(rowBox, box) >= 0.15 || (center >= box.yMin && center <= box.yMax);
  });
}

function bandsHaveAmbiguousOverlap(bands) {
  const sorted = (bands || []).slice().sort(compareRows);
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const upper = sorted[i].bbox;
    const lower = sorted[i + 1].bbox;
    if (verticalOverlapRatio(upper, lower) >= 0.08) return true;
    if (verticalGap(upper, lower) <= 2) return true;
  }
  return false;
}

function rowHasLocalSupport(child, parent, allowance) {
  let supported = 0;
  for (const childStroke of child.strokes || []) {
    const childBox = childStroke.canvasBbox;
    const local = (parent.strokes || []).some((parentStroke) => {
      const parentBox = parentStroke.canvasBbox;
      return horizontalGap(childBox, parentBox) <= allowance ||
        horizontalOverlapRatio(childBox, parentBox) >= 0.2;
    });
    if (local) supported += 1;
  }
  return child.strokes.length > 0 && supported / child.strokes.length >= 0.6;
}

function rowHasTallOperatorStroke(row, medianHeight) {
  return (row?.strokes || []).some((stroke) => (
    strokeHeight(stroke) >= Math.max(48, medianHeight * 1.55)
  ));
}

function rowHasLocalHorizontalBridge(row, child) {
  const childBox = child?.bbox;
  if (!childBox) return false;
  const childWidth = Math.max(1, bboxWidth(childBox));
  return (row?.strokes || []).some((stroke) => {
    const box = stroke.canvasBbox;
    if (!box || !isHorizontalStroke(stroke)) return false;
    const width = bboxWidth(box);
    if (width < childWidth * 0.6) return false;
    return horizontalOverlapRatio(box, childBox) >= 0.45 ||
      horizontalGap(box, childBox) <= Math.max(18, childWidth * 0.2);
  });
}

function rowLooksLikeEquationWithEquals(row) {
  const horizontals = (row?.strokes || [])
    .filter((stroke) => {
      const box = stroke.canvasBbox;
      if (!box || !isHorizontalStroke(stroke)) return false;
      const width = bboxWidth(box);
      const height = bboxHeight(box);
      return width >= 18 && width <= 90 && height <= 14;
    })
    .sort((a, b) => centerY(a.canvasBbox) - centerY(b.canvasBbox));

  for (let i = 0; i < horizontals.length; i += 1) {
    for (let j = i + 1; j < horizontals.length; j += 1) {
      const upper = horizontals[i].canvasBbox;
      const lower = horizontals[j].canvasBbox;
      const gap = verticalGap(upper, lower);
      if (gap > 22) continue;
      if (horizontalOverlapRatio(upper, lower) < 0.65) continue;
      return true;
    }
  }
  return false;
}

function hasWideFractionBar(candidate) {
  if (!candidate?.strokes?.length || !candidate.tightBbox) return false;
  const candidateWidth = Math.max(1, bboxWidth(candidate.tightBbox));
  return candidate.strokes.some((stroke) => {
    const box = stroke.canvasBbox;
    if (!box || !isHorizontalStroke(stroke)) return false;
    return bboxWidth(box) >= Math.max(90, candidateWidth * 0.42);
  });
}

function rowLooksLikeOperationAnnotation(row) {
  const box = row?.bbox;
  const strokes = row?.strokes || [];
  if (!box || strokes.length < 3) return false;
  const width = bboxWidth(box);
  const height = bboxHeight(box);
  if (width < 110 || height > 80) return false;

  const horizontalMarks = strokes.filter((stroke) => {
    const strokeBox = stroke.canvasBbox;
    if (!strokeBox || !isHorizontalStroke(stroke)) return false;
    const strokeWidthValue = bboxWidth(strokeBox);
    const strokeHeightValue = bboxHeight(strokeBox);
    return strokeWidthValue >= 16 &&
      strokeWidthValue <= Math.max(90, width * 0.42) &&
      strokeHeightValue <= 16;
  });
  if (horizontalMarks.length < 2) return false;

  const centers = horizontalMarks.map((stroke) => centerX(stroke.canvasBbox));
  return Math.max(...centers) - Math.min(...centers) >= Math.max(70, width * 0.35);
}

function hasNearbyCompactFractionBridge(candidate, rows) {
  if (!candidate?.strokes?.length || !rows || rows.length < 2 || rows.length > 3) return false;

  for (const stroke of candidate.strokes) {
    const bar = stroke.canvasBbox;
    if (!bar || !isHorizontalStroke(stroke)) continue;
    const width = bboxWidth(bar);
    const height = Math.max(1, bboxHeight(bar));
    if (width < 12 || width / height < 3) continue;

    const barY = centerY(bar);
    let supportedAbove = false;
    let supportedBelow = false;
    let nearestAboveGap = Infinity;
    let nearestBelowGap = Infinity;

    for (const other of candidate.strokes) {
      if (other === stroke || !other.canvasBbox) continue;
      const box = other.canvasBbox;
      if (!barHasLocalSupport(bar, box, 0.16)) continue;
      const otherY = centerY(box);
      if (otherY < barY - height * 0.25) {
        supportedAbove = true;
        nearestAboveGap = Math.min(nearestAboveGap, Math.max(0, bar.yMin - box.yMax));
      }
      if (otherY > barY + height * 0.25) {
        supportedBelow = true;
        nearestBelowGap = Math.min(nearestBelowGap, Math.max(0, box.yMin - bar.yMax));
      }
    }

    if (!supportedAbove || !supportedBelow) continue;
    const aboveGap = Math.max(4, nearestAboveGap);
    const belowGap = Math.max(4, nearestBelowGap);
    if (Math.max(aboveGap, belowGap) > Math.max(24, height * 4)) continue;
    if (Math.max(aboveGap, belowGap) / Math.min(aboveGap, belowGap) <= 4.5) return true;
  }

  return false;
}

function hasCompactStackedFractionColumn(candidate, rows) {
  if (!candidate?.strokes?.length || !rows || rows.length < 2 || rows.length > 3) return false;

  const sortedRows = rows.slice().sort(compareRows);
  const candidateBox = candidate.tightBbox || computeTightBbox(candidate.strokes);
  const candidateWidth = bboxWidth(candidateBox);
  const candidateHeight = bboxHeight(candidateBox);
  const medianHeight = median(candidate.strokes.map(strokeHeight)) || 1;
  if (candidateHeight > Math.max(120, medianHeight * 5.8)) return false;

  for (let index = 0; index < sortedRows.length - 1; index += 1) {
    const upper = sortedRows[index];
    const upperWidth = bboxWidth(upper.bbox);
    const upperHeight = bboxHeight(upper.bbox);

    if ((upper.strokes || []).length > 3) continue;
    if (upperWidth > Math.max(82, medianHeight * 3.5)) continue;
    if (upperWidth > candidateWidth * 0.42) continue;

    for (let lowerIndex = index + 1; lowerIndex < sortedRows.length; lowerIndex += 1) {
      const lower = sortedRows[lowerIndex];
      const lowerHeight = bboxHeight(lower.bbox);
      const gap = verticalGap(upper.bbox, lower.bbox);

      if (centerY(upper.bbox) >= centerY(lower.bbox)) continue;
      if (gap > Math.max(28, medianHeight * 1.45)) continue;
      if (upperHeight > lowerHeight * 1.05 && upperHeight > medianHeight * 1.6) continue;

      const alignedLowerInk = (lower.strokes || []).some((stroke) => {
        const box = stroke.canvasBbox;
        if (!box) return false;
        if (centerY(box) <= centerY(upper.bbox)) return false;
        const centerDistance = Math.abs(centerX(box) - centerX(upper.bbox));
        const overlaps = horizontalOverlapRatio(upper.bbox, box) >= 0.35;
        const centered = centerDistance <= Math.max(22, Math.min(upperWidth, bboxWidth(box)) * 0.9);
        if (!overlaps && !centered) return false;
        return bboxWidth(box) <= Math.max(92, upperWidth * 2.4, medianHeight * 4.2);
      });

      if (alignedLowerInk) return true;
    }
  }

  return false;
}

function barHasLocalSupport(bar, box, minOverlapRatio) {
  if (!bar || !box) return false;
  if (horizontalOverlapRatio(bar, box) >= minOverlapRatio) return true;
  const barWidth = bboxWidth(bar);
  const gap = horizontalGap(bar, box);
  if (gap > Math.max(18, barWidth * 0.65)) return false;

  const barCenter = (bar.xMin + bar.xMax) / 2;
  const boxCenter = (box.xMin + box.xMax) / 2;
  const centerOffset = Math.abs(barCenter - boxCenter);
  return centerOffset <= Math.max(48, barWidth * 1.7);
}

function mostlyHorizontal(row) {
  const strokes = row?.strokes || [];
  if (strokes.length === 0) return false;
  return strokes.filter(isHorizontalStroke).length / strokes.length >= 0.75;
}

function isHorizontalStroke(stroke) {
  return stroke?.canvasBbox && strokeWidth(stroke) / strokeHeight(stroke) >= 3;
}

/**
 * Detect strokes that look like diagonal strike-through (crossed-out) marks.
 * A crossed-out stroke is typically long, thin, and diagonal, crossing through
 * the bounding boxes of multiple other strokes in the same candidate.
 */
export function detectCrossedOutStrokes(strokes) {
  const usable = (strokes || []).filter((stroke) => stroke?.canvasBbox);
  if (usable.length < 3) return [];

  const medianHeight = median(usable.map(strokeHeight)) || 1;
  const crossedOut = [];

  for (const stroke of usable) {
    const box = stroke.canvasBbox;
    const width = bboxWidth(box);
    const height = bboxHeight(box);
    const length = Math.sqrt(width * width + height * height);
    if (length < Math.max(30, medianHeight * 1.5)) continue;
    if (width < height * 1.2) continue;

    const aspectRatio = width / Math.max(1, height);
    if (aspectRatio < 1.2 || aspectRatio > 12) continue;

    const angle = Math.atan2(height, width);
    if (angle < 0.15 || angle > 1.4) continue;

    const others = usable.filter((other) => other !== stroke);
    const crossedCount = others.filter((other) => {
      const otherBox = other.canvasBbox;
      const overlap = Math.max(0, Math.min(box.xMax, otherBox.xMax) - Math.max(box.xMin, otherBox.xMin));
      const overlapRatio = overlap / Math.max(1, Math.min(width, bboxWidth(otherBox)));
      return overlapRatio >= 0.3;
    }).length;

    if (crossedCount >= 2) {
      crossedOut.push(stroke);
    }
  }

  return crossedOut;
}

/**
 * Detect small isolated scratch annotations positioned above the main content.
 * These are short strokes or small groups of strokes that sit above the main
 * equation line with a vertical gap, often representing mental arithmetic
 * notes like "+1" or scratch calculations.
 */
export function detectScratchAnnotations(strokes) {
  const usable = (strokes || []).filter((stroke) => stroke?.canvasBbox);
  if (usable.length < 4) return [];

  const sorted = usable.slice().sort(compareStrokeCenterY);
  const medianHeight = median(usable.map(strokeHeight)) || 1;
  const overallBox = computeTightBbox(usable);
  if (!overallBox) return [];

  const topThreshold = overallBox.yMin + medianHeight * 0.8;
  const topStrokes = sorted.filter((stroke) => centerY(stroke.canvasBbox) < topThreshold);
  if (topStrokes.length === 0 || topStrokes.length === usable.length) return [];

  const topBox = computeTightBbox(topStrokes);
  if (!topBox) return [];
  const topWidth = bboxWidth(topBox);
  const topHeight = bboxHeight(topBox);

  if (topWidth > bboxWidth(overallBox) * 0.5) return [];
  if (topStrokes.length > 4) return [];
  if (topHeight > medianHeight * 2.5) return [];

  const remaining = usable.filter((stroke) => !topStrokes.includes(stroke));
  if (remaining.length < 2) return [];
  const remainingBox = computeTightBbox(remaining);
  if (!remainingBox) return [];

  const gap = remainingBox.yMin - topBox.yMax;
  if (gap < 4 || gap > medianHeight * 1.5) return [];

  if (horizontalOverlapRatio(topBox, remainingBox) < 0.2) return [];

  return topStrokes;
}

export function detectEnclosingAnnotationStrokes(strokes) {
  const usable = (strokes || []).filter((stroke) => stroke?.canvasBbox);
  if (usable.length < 4) return [];

  const medianHeight = median(usable.map(strokeHeight)) || 1;
  const medianWidth = median(usable.map(strokeWidth)) || 1;
  const out = [];

  for (const stroke of usable) {
    const box = stroke.canvasBbox;
    const width = bboxWidth(box);
    const height = bboxHeight(box);
    if (width < Math.max(90, medianWidth * 2.8)) continue;
    if (height < Math.max(70, medianHeight * 2.2)) continue;
    const aspect = width / Math.max(1, height);
    if (aspect < 0.45 || aspect > 2.6) continue;
    if (isHorizontalStroke(stroke) || strokeWidth(stroke) / Math.max(1, strokeHeight(stroke)) < 0.45) continue;

    const enclosed = usable.filter((other) => {
      if (other === stroke || !other.canvasBbox) return false;
      const otherBox = other.canvasBbox;
      const centerInside = centerX(otherBox) >= box.xMin && centerX(otherBox) <= box.xMax &&
        centerY(otherBox) >= box.yMin && centerY(otherBox) <= box.yMax;
      if (!centerInside) return false;
      return bboxWidth(otherBox) <= width * 0.55 && bboxHeight(otherBox) <= height * 0.72;
    });
    if (enclosed.length < 3) continue;

    const enclosedBox = computeTightBbox(enclosed);
    if (!enclosedBox) continue;
    if (horizontalOverlapRatio(box, enclosedBox) < 0.85) continue;
    if (verticalOverlapRatio(box, enclosedBox) < 0.85) continue;
    out.push(stroke);
  }

  return out;
}

export function detectSeparatedTopAnnotationStrokes(strokes) {
  const usable = (strokes || []).filter((stroke) => stroke?.canvasBbox);
  if (usable.length < 8) return [];

  const sorted = usable.slice().sort(compareStrokeCenterY);
  const medianHeight = median(usable.map(strokeHeight)) || 1;
  const medianWidth = median(usable.map(strokeWidth)) || 1;
  const splitIndexes = [];

  for (let index = 0; index < sorted.length - 1; index += 1) {
    const gap = centerY(sorted[index + 1].canvasBbox) - centerY(sorted[index].canvasBbox);
    const topCount = index + 1;
    const restCount = sorted.length - topCount;
    if (topCount < 2 || topCount > 8 || restCount < 5) continue;
    if (gap >= Math.max(8, medianHeight * 0.65)) {
      splitIndexes.push(index + 1);
    }
  }
  if (splitIndexes.length === 0) return [];

  const overallBox = computeTightBbox(usable);
  if (!overallBox) return [];

  for (const splitIndex of splitIndexes) {
    const topStrokes = sorted.slice(0, splitIndex);
    const remaining = sorted.slice(splitIndex);
    const topBox = computeTightBbox(topStrokes);
    const remainingBox = computeTightBbox(remaining);
    if (!topBox || !remainingBox) continue;
    if (topBox.yMax > remainingBox.yMin + medianHeight * 0.2) continue;

    const topWidth = bboxWidth(topBox);
    const topHeight = bboxHeight(topBox);
    const topCoverage = topStrokes.reduce((sum, stroke) => sum + strokeWidth(stroke), 0) / Math.max(1, topWidth);
    const verticalGap = remainingBox.yMin - topBox.yMax;
    const sparseWideMark = (
      topStrokes.length <= 4 &&
      topWidth >= Math.max(240, medianWidth * 8) &&
      topHeight <= medianHeight * 2.2 &&
      topCoverage <= 0.28 &&
      verticalGap >= medianHeight * 1.0
    );

    const hasLongStrike = topStrokes.some((stroke) => (
      strokeWidth(stroke) >= Math.max(70, medianWidth * 2.2) &&
      strokeHeight(stroke) <= medianHeight * 0.35
    ));
    const hasLooseEnclosingStroke = topStrokes.some((stroke) => {
      const width = strokeWidth(stroke);
      const height = strokeHeight(stroke);
      const aspect = width / Math.max(1, height);
      return (
        height >= medianHeight * 2.6 &&
        width >= medianWidth * 1.4 &&
        aspect >= 0.35 &&
        aspect <= 1.3
      );
    });
    const crossedScratchMark = (
      topStrokes.length <= 8 &&
      hasLongStrike &&
      hasLooseEnclosingStroke &&
      topHeight <= bboxHeight(overallBox) * 0.48
    );

    if (sparseWideMark || crossedScratchMark) return topStrokes;
  }

  return [];
}

export function detectTinyIsolatedScratchStrokes(strokes) {
  const usable = (strokes || []).filter((stroke) => stroke?.canvasBbox);
  if (usable.length === 0) return [];

  return usable.filter((stroke) => {
    const width = strokeWidth(stroke);
    const height = strokeHeight(stroke);
    if (width > 10 || height > 10) return false;

    return !usable.some((other) => {
      if (other === stroke || !other.canvasBbox) return false;
      const horizontalGap = Math.max(
        0,
        Math.max(other.canvasBbox.xMin, stroke.canvasBbox.xMin) -
          Math.min(other.canvasBbox.xMax, stroke.canvasBbox.xMax)
      );
      const verticalGap = Math.max(
        0,
        Math.max(other.canvasBbox.yMin, stroke.canvasBbox.yMin) -
          Math.min(other.canvasBbox.yMax, stroke.canvasBbox.yMax)
      );
      return Math.hypot(horizontalGap, verticalGap) <= 32;
    });
  });
}

/**
 * Check if a candidate contains crossed-out or scratch annotation strokes
 * and return a penalty score if it does.
 */
function annotationPenalty(candidate) {
  const strokes = candidate?.strokes || [];
  if (strokes.length < 3) return 0;

  const crossedOut = detectCrossedOutStrokes(strokes);
  const scratch = detectScratchAnnotations(strokes);
  let penalty = 0;

  if (crossedOut.length > 0) {
    const nonCrossedCount = strokes.length - crossedOut.length;
    if (nonCrossedCount < 2) {
      penalty += 6.0;
    } else {
      penalty += 2.5;
    }
  }

  if (scratch.length > 0) {
    const nonScratchCount = strokes.length - scratch.length;
    if (nonScratchCount < 2) {
      penalty += 5.0;
    } else {
      penalty += 1.5;
    }
  }

  return penalty;
}

function hasCloserUpperNeighbor(childIndex, parentIndex, rows, gapToParent) {
  if (centerY(rows[childIndex].bbox) >= centerY(rows[parentIndex].bbox)) return false;
  for (let i = 0; i < rows.length; i += 1) {
    if (i === childIndex || i === parentIndex) continue;
    if (centerY(rows[i].bbox) >= centerY(rows[childIndex].bbox)) continue;
    const neighborGap = verticalGap(rows[i].bbox, rows[childIndex].bbox);
    if (neighborGap <= gapToParent * 1.2 + 8) return true;
  }
  return false;
}

function rowMedianHeight(row) {
  return median((row?.strokes || []).map(strokeHeight));
}

function mergeRows(target, source) {
  target.bbox = bboxUnion(target.bbox, source.bbox);
  target.strokes = uniqueStrokes(target.strokes.concat(source.strokes || []));
  target.detections = (target.detections || []).concat(source.detections || []);
}

function assignConflicts(candidates) {
  for (const candidate of candidates) candidate.conflicts = [];
  for (let i = 0; i < candidates.length; i += 1) {
    for (let j = i + 1; j < candidates.length; j += 1) {
      if (!candidatesOverlap(candidates[i], candidates[j])) continue;
      candidates[i].conflicts.push(candidates[j].candidateId);
      candidates[j].conflicts.push(candidates[i].candidateId);
    }
  }
}

function childLinesCoverParent(parent, children) {
  if (!parent?.strokeIds?.length || children.length <= 1) return false;
  const parentIds = new Set(parent.strokeIds);
  const covered = new Set();
  for (const child of children) {
    if (!strokeSetContains(parent, child)) continue;
    for (const strokeId of child.strokeIds) {
      if (parentIds.has(strokeId)) covered.add(strokeId);
    }
  }
  return covered.size === parentIds.size;
}

function strokeSetContains(container, contained) {
  const ids = new Set(container.strokeIds || []);
  return (contained.strokeIds || []).every((strokeId) => ids.has(strokeId));
}

function candidatesOverlap(a, b) {
  const ids = new Set(a.strokeIds || []);
  return (b.strokeIds || []).some((strokeId) => ids.has(strokeId));
}

function fallbackCover(entries) {
  const selected = [];
  const covered = new Set();
  const sorted = entries.slice().sort((a, b) => b.selectionScore - a.selectionScore);
  for (const entry of sorted) {
    if (entry.candidate.strokeIds.some((id) => covered.has(id))) continue;
    selected.push(entry);
    for (const id of entry.candidate.strokeIds) covered.add(id);
  }
  return selected;
}

function partitionCandidateIds(candidates) {
  const partitions = {};
  for (const candidate of candidates) {
    for (const profile of candidate.profiles || []) {
      if (!partitions[profile]) partitions[profile] = [];
      partitions[profile].push(candidate.candidateId);
    }
  }
  return partitions;
}

function sortCandidates(candidates) {
  return (candidates || []).slice().sort((a, b) => (
    (a.tightBbox?.yMin ?? 0) - (b.tightBbox?.yMin ?? 0) ||
    (a.tightBbox?.xMin ?? 0) - (b.tightBbox?.xMin ?? 0) ||
    String(a.candidateId).localeCompare(String(b.candidateId))
  ));
}

function strokeSetKeyFor(strokes) {
  return uniqueStrings((strokes || []).map((stroke) => String(stroke.id))).sort().join('|');
}

function uniqueStrokes(strokes) {
  const seen = new Set();
  const out = [];
  for (const stroke of strokes || []) {
    const id = String(stroke?.id ?? out.length);
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(stroke);
  }
  return out;
}

function uniqueStrings(values) {
  return [...new Set((values || []).map(String))];
}

function bboxesOverlap(a, b) {
  return !!a && !!b && a.xMin <= b.xMax && a.xMax >= b.xMin &&
    a.yMin <= b.yMax && a.yMax >= b.yMin;
}

function bboxUnion(a, b) {
  return {
    xMin: Math.min(a.xMin, b.xMin),
    yMin: Math.min(a.yMin, b.yMin),
    xMax: Math.max(a.xMax, b.xMax),
    yMax: Math.max(a.yMax, b.yMax)
  };
}

function bboxWidth(box) {
  return box ? Math.max(1, box.xMax - box.xMin) : 1;
}

function bboxHeight(box) {
  return box ? Math.max(1, box.yMax - box.yMin) : 1;
}

function strokeHeight(stroke) {
  return stroke?.canvasBbox ? bboxHeight(stroke.canvasBbox) : 1;
}

function strokeWidth(stroke) {
  return stroke?.canvasBbox ? bboxWidth(stroke.canvasBbox) : 1;
}

function verticalGap(a, b) {
  if (!a || !b) return Infinity;
  if (a.yMin <= b.yMax && a.yMax >= b.yMin) return 0;
  return a.yMax < b.yMin ? b.yMin - a.yMax : a.yMin - b.yMax;
}

function horizontalGap(a, b) {
  if (!a || !b) return Infinity;
  if (a.xMin <= b.xMax && a.xMax >= b.xMin) return 0;
  return a.xMax < b.xMin ? b.xMin - a.xMax : a.xMin - b.xMax;
}

function centerY(box) {
  return (box.yMin + box.yMax) / 2;
}

function centerX(box) {
  return (box.xMin + box.xMax) / 2;
}

function compareRows(a, b) {
  return a.bbox.yMin - b.bbox.yMin || a.bbox.xMin - b.bbox.xMin;
}

function compareStrokeCenterY(a, b) {
  return centerY(a.canvasBbox) - centerY(b.canvasBbox) ||
    a.canvasBbox.xMin - b.canvasBbox.xMin;
}

function strokeTime(stroke) {
  const start = Number(stroke?.startTime);
  return Number.isFinite(start) ? start : 0;
}

function median(values) {
  const usable = (values || []).filter((value) => Number.isFinite(Number(value))).map(Number);
  if (usable.length === 0) return 0;
  usable.sort((a, b) => a - b);
  return usable[Math.floor(usable.length / 2)];
}

function bracesAreBalanced(latex) {
  let depth = 0;
  for (const char of String(latex || '')) {
    if (char === '{') depth += 1;
    if (char === '}') depth -= 1;
    if (depth < 0) return false;
  }
  return depth === 0;
}

function extractMathSymbols(latex) {
  const symbols = new Set();
  for (const match of String(latex || '').matchAll(/\\[A-Za-z]+|[A-Za-z0-9]/g)) {
    const token = match[0].replace(/^\\/, '');
    if (!['left', 'right', 'frac', 'sqrt', 'cdot', 'times'].includes(token)) {
      symbols.add(token);
    }
  }
  return symbols;
}

function normalizeEvidenceScores(scores) {
  if (!scores) return new Map();
  const entries = scores instanceof Map ? scores.entries() : Object.entries(scores);
  const normalized = new Map();
  for (const [candidateId, score] of entries) {
    const numericScore = Number(score);
    if (Number.isFinite(numericScore)) {
      normalized.set(candidateId, numericScore);
    }
  }
  return normalized;
}
